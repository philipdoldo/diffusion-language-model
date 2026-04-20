import torch
import argparse
import yaml
import os
import math
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from model import DiT, DiTConfig
from dataloader import ShardDataLoader

"""
We define a forward noising process using a CTMC. This CTMC has a time-dependent rate matrix, but it makes a significant
simplifying assumption on its structure, namely, that the rate matrix can be written as sigma(t) * Q where Q is a fixed
matrix and sigma(t) is a scalar function of time (which we call the noise schedule). This is in addition to the sparisty
constraints that we're enforcing as well, so this can be done at the token level (rather than the sequence level).

At t=0 we want pure data sampled from p_data and at t=T we want pure noise sampled from p_noise. Theorem 3.6 in the SEDD
paper gives an ELBO where
    -log(p_0^{theta}(x_0)) <= L_{DWDSE} + D_{KL}( p_{T|0}(.|x_0) || p_noise )
yet in Algorithm 1 we notice the absence of the KL divergence term. This is because the noise schedule sigma(t) is chosen
in a specific way to make the KL divergence term empirically close to zero in practice -- I'll elaborate on this: Because
of how the rate matrix is simplified, the transition probabilities are in a very similar form to those we'd obtain for a
time-homogeneous CTMC where instead of p_t = exp(Qt) p_0 we get p_t = exp(sigma_bar(t) Q) p_0 where 
    sigma_bar(t) := int_{0}^{t} sigma(s) ds
so you can view sigma_bar(t) as a transformation on the time of a time-homogeneous CTMC, so the CTMC approximately converges
to its stationary distribution when sigma_bar(t) is really big. For uniform diffusion, Q was chosen to make its stationary
distribution pure noise, that is, Q = (11^T - N*I)/N where 1 is a column vector of ones and I is the identity matrix. Solving
Qp = 0 results in every entry of p being 1/N, so we see that the uniform distribution is indeed the stationary distribution
for a time-homogeneous CTMC with rate matrix given by Q (we just plugged in a constant p into the Kolmogorov forward equation
and solved for p). So basically, we want to choose our noise transformation sigma_bar(t) in a way such that sigma_bar(T)
is big enough to decently approximate the uniform distribution p_noise to make D_{KL}( p_{T|0}(.|x_0) || p_noise ) ~= 0 in
practice. This is one approximation being made, but we actually make another approximation (following the SEDD paper).
    When defining sigma_bar(t) in practice, we don't always exactly use sigma_bar(t) = int_{0}^{t} sigma(s) ds, instead in
some cases (e.g., the geometric noise defined below) we have sigma_bar(t) ~= int_{0}^{t} sigma(s) ds where sigma_bar(0) =/= 0
but sigma_bar(0) ~= 0. However, when taking a column of the transition matrix to sample from, they use
    p_{t|0}(.|x_0) = exp(sigma_bar(t) Q)_{x_0}
which satifies the KFE but only satisfies the appropriate initial condition (p_{0|0} = I) if sigma_bar(0) = 0. The KFE for 
p_{t|0} is
    d/dt[p_{t|0}] = sigma(t) Q p_{t|0}
(where in practice, sigma(t) = d/dt[sigma_bar(t)], even if sigma_bar(t) is only approximately int_{0}^{t} sigma(s) ds) and
the solution is:
    p_{t|0} = exp(sigma_bar(t) Q) c
where c is a constant vector to be determined by the initial condition p_{0|0} = I
    p_{0|0} = exp(sigma_bar(0) Q) c = I
==> c = exp(-sigma_bar(0) Q)
==> p_{t|0} = exp(sigma_bar(t) Q) exp(-sigma_bar(0) Q) = exp( (sigma_bar(t) - sigma_bar(0)) Q)
If sigma_bar(0) were 0, then we'd recover the form used in Algorithm 1 of the SEDD paper, but if sigma_bar(0) ~= 0 then the form
in the paper approximates this shifted solution. 
    Presumably, we also want the noise schedule to gradually increase to pure noise over the course of traversing forward in
time across the interval [0,T] such that the process doesn't get too noisy until time is close to T. Choosing a large value
for sigma_bar(0), for example, would make the process too noisy too quickly and thus the task of learning to denoise becomes
unreasonably difficult. Having sigma_bar(0) ~= 0 and not allowing our noise schedule to increase too quickly makes it so that
p_{s|0} ~= I for small s > 0 so that p_s = E_{x_0 ~ p_data}[p_{s|0}(.|x_0)] ~= p_data for small s. In principle, it should be
easy to precompute the shift in closed form, but I'm not going to bother to keep things as simple as possible.
"""

class GeometricNoise:
    """
    Takes two positive floats as input: `sigma_min` and `sigma_max` and defines
        sigma_bar(t) = sigma_min^(1-t) * sigma_max^t
    `sigma_min` is positive but close enough to zero to get exp(-sigma_bar(0) Q) ~= I, so there is almost no noise early on
    `sigma_max` is chosen to be big enough so that at t=T the forward process is close to the stationary distribution.
    We assume T = 1, so we only allow times `t` in [0, 1]

    One could opt to make sigma_min and sigma_max learnable parameters, but I'm not going to do that for simplicity
    """

    def __init__(self, sigma_min=1e-4, sigma_max=20):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma_bar(self, t):
        """
        `t` has shape (batch_size,)
        """
        if (t > 1).any() or (t < 0).any():
            raise ValueError(f"Expected t in [0,1], got {t=}")
        return self.sigma_min**(1-t) * self.sigma_max**t
    
    def sigma(self, t):
        """
        This is the derivative of sigma_bar(t). In practice, we define sigma_bar(t) first and then take its derivative to get
        sigma(t) rather than starting with sigma(t) and computing sigma_bar(t) = int_{0}^{t} sigma(s) ds. There isn't a principled
        reason for this, this just happens to be what was done for this geometric noise schedule used in the SEDD paper.
            `t` has shape (batch_size,)
        """
        if (t > 1).any() or (t < 0).any():
            raise ValueError(f"Expected t in [0,1], got {t=}")
        return self.sigma_min**(1-t) * self.sigma_max**t * (torch.log(torch.tensor(self.sigma_max)) - torch.log(torch.tensor(self.sigma_min)))

"""
For uniform diffusion, we are using Q = (11^T - N*I)/N = (1/N)*11^T - I and (1/N)*11^T is a projection matrix, so the matrix
exponential is very easy to compute. This makes it so that 
    exp(sigma_bar(t)*Q) = (1 - exp(-sigma_bar(t)))*(1/N)*11^T + exp(-sigma_bar(t))*I
which defines our transition probabilities p_{t|0}. In practice, we are going to start at some initial state x_0^{i} (a given token
at sequence position i) and we'll care about only the corresponding column of our transition matrix given by p_{t|0}(.|x_0^{i}).
"""

class UniformCTMC:

    def __init__(self, config):
        self.noise = GeometricNoise(sigma_min=config["UniformCTMC"]["sigma_min"], sigma_max=config["UniformCTMC"]["sigma_max"])
        self.N = config["model"]["vocab_size"] # the rate and transition matrices are N-by-N (this is all at the token level rather than the sequence level)

    def transition(self, cols, t):
        """
        For each batch and sequence position, get a column of the transition matrix p_{t|0}, specifically, get p_{t|0}(.|col) where col is a token id
        We're just getting a column of 
            exp(sigma_bar(t)*Q) = (1 - exp(-sigma_bar(t)))*(1/N)*11^T + exp(-sigma_bar(t))*I
        which is just a vector of ones multiplied by the scalar (1 - exp(-sigma_bar(t)))*(1/N) plus the scalar exp(-sigma_bar(t)) in the col'th position

        Really we need to generate a probability vector over tokens for every single position in our sequence (for the same time t), also we want to do
        this for a batch of sequences, so we should have:
            `t` shape (batch_size,)
            `cols` shape (batch_size, seq_len) -- these are token sequences sampled from p_data (remember p_0 ~= p_data)
        outputs `p` with shape (batch_size, seq_len, N) where N is the vocab size. The idea is that we can use this output to get a sample for batches of
        the sequence at time t (x_t) which has shape (batch_size, seq_len) by independently sampling a token for each sequence position at each batch index
        """
        batch_size = t.shape[0]
        seq_len = cols.shape[1]
        if len(t.shape) > 1 or len(cols.shape) > 2 or t.shape[0] != cols.shape[0]:
            raise ValueError(f"{t.shape=}, {cols.shape=}")
        
        sigma_bar = self.noise.sigma_bar(t) # (batch_size,)
        c = torch.exp(-sigma_bar) # (batch_size,)
        coeff = ((1 - c[:, None, None]) / self.N) # (batch_size, 1, 1)
        p = coeff.expand(batch_size, seq_len, self.N).clone() # (batch_size, seq_len, N)

        # We need to add c * I (i.e., add exp(-sigma_bar(t))*I)
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         for k in range(1):  # cols[..., None] has shape (batch_size, seq_len, 1)
        #             p[i][j][cols[i][j][k]] += c[i][j][k] # assuming c has shape (batch_size, seq_len, 1), which doesn't happen without .expand()
        p.scatter_add_(-1, cols[..., None].long(), c[:, None, None].expand(batch_size, seq_len, 1))
        return p # (batch_size, seq_len, N)


def sample_categorical(p):
    """
    `p` has shape (batch_size, seq_len, vocab_size) and is basically p_{t|0}(.|x_0^{i}) for each sequence index i and each batch index
    This function samples a token in {0, ..., vocab_size-1} for each batch index and sequence position resulting in a tensor with
    shape (batch_size, seq_len). This is effectively sampling x_t^{i} ~  p_{t|0}(.|x_0^{i}) for each sequence position i and batch index.

    Motivated by https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/catsample.py#L10

    ---
    Notes:
    Let p in R^n be a probability vector and let G_1, ..., G_n ~ Gumbel(0, 1) iid
    We can use the fact that
        argmax_{k in {1, ..., n}}( log(p_k) + G_k ) ~ Categorical(p)
    However, we can equivalently let U_1, ..., U_n ~ Unif([0,1]) iid and then
        argmax_{k in {1, ..., n}}( p_k / (-log(U_k)) ) ~ Categorical(p)
    Note that if U ~ Unif([0,1]), then -log(U) ~ Exp(1) and if E ~ Exp(1), then -log(E) ~ Gumbel(0, 1), so
        argmax_{k}( p_k / (-log(U_k)) ) = argmax_{k}( log(p_k) - log(-log(U_k)) ) = argmax_{k}( log(p_k) + G_k )
    This way should actually be a little faster since it should be one fewer log computation since under the hood
    pytorch generates gumbel samples by sampling U ~ Unif([0,1]) and then doing -log(-log(U)) ~ Gumbel(0, 1)

    In practice we add a small tolerance inside and outside of the log to avoid log(0) or log(1) (the latter would cause division by 0)
    """
    eps = 1e-10
    exp_norm = eps - torch.rand_like(p + eps).log() # -log(U) ~ Exp(1) if U ~ Unif([0,1])
    return (p / exp_norm).argmax(dim=-1) # shape (batch_size, seq_len)


def loss_DWDSE(log_score_model, x_0, t, ctmc):
    """
    The Diffusion Weighted Denoising Score Entropy loss L_{DWDSE} from the SEDD paper https://arxiv.org/pdf/2310.16834 (see Algorithm 1)

    `log_score_model` is our neural network that outputs `log_scores`
    `x_0` has shape (batch_size, seq_len) and is a sequence of token ids sampled from p_data
    `t` has shape (batch_size,)
    `ctmc` is an instance of our UniformCTMC class, we'll use it to compute `p` and `sigma`

    `log_scores` has shape (batch_size, seq_len, vocab_size) where log_scores corresponding to the original token id are 0 (see model.py)
    `p` has shape (batch_size, seq_len, vocab_size) and represents p_{t|0}(.|x_0^{i}) for each sequence position i and each batch index
    `sigma` has shape (batch_size,) and is simply sigma(t) corresponding to the noise schedule being used
    """
    p = ctmc.transition(cols=x_0, t=t)  # (batch_size, seq_len, vocab_size)
    x_t = sample_categorical(p)  # (batch_size, seq_len)
    log_scores = log_score_model(token_ids=x_t, t=t)  # (batch_size, seq_len, vocab_size)
    sigma_bar = ctmc.noise.sigma_bar(t)  # (batch_size,)
    dsigma = ctmc.noise.sigma(t)  # (batch_size,)
    N = ctmc.N

    esigm1 = torch.where(
        sigma_bar[:, None] < 0.5,
        torch.expm1(sigma_bar[:, None]),
        sigma_bar[:, None].exp() - 1
    )  # (batch_size, seq_len)

    # ratio = 1 - N / (esigm1 + N), the true score ratio when x_t == x_0
    ratio = 1 - N / (esigm1 + N)  # (batch_size, seq_len)

    # pos_term: (1/N) * sum_{y != x_t} s_theta(x_t)_y
    scores = log_scores.exp()  # (batch_size, seq_len, vocab_size)
    pos_term = scores.mean(dim=-1) - scores.gather(-1, x_t[..., None]).squeeze(-1) / N  # (batch_size, seq_len)

    # neg_term: analytically computed using structure of uniform transition matrix
    log_score_sum = log_scores.mean(dim=-1) - log_scores.gather(-1, x_t[..., None]).squeeze(-1) / N  # (batch_size, seq_len)
    no_move = (x_t == x_0.long())  # (batch_size, seq_len)
    neg_term = torch.where(
        no_move,
        ratio * log_score_sum,
        log_scores.gather(-1, x_0[..., None].long()).squeeze(-1) / esigm1 + log_score_sum
    )  # (batch_size, seq_len)

    # const: K(r) term, makes loss non-negative, no gradient w.r.t. theta
    const = torch.where(
        no_move,
        (N - 1) / N * ratio * (ratio.log() - 1),
        ((-ratio.log() - 1) / ratio - (N - 2)) / N
    ).detach()  # (batch_size, seq_len)

    loss = pos_term - neg_term + const  # (batch_size, seq_len)
    loss = (dsigma[:, None] * loss).sum(dim=-1)  # (batch_size,)
    loss = loss.mean()  # scalar
    return loss
    # p = ctmc.transition(cols=x_0, t=t) # (batch_size, seq_len, vocab_size)
    # x_t = sample_categorical(p) # (batch_size, seq_len)
    # log_scores = log_score_model(token_ids=x_t, t=t) # (batch_size, seq_len, vocab_size)
    # sigma = ctmc.noise.sigma(t) # (batch_size,)
    # print(f"{t=}")
    # print(f"{sigma=}")
    # print(f"{log_scores=}")
    # print(f"{x_t=}")
    # print(f"{p=}")
    # vocab_size = p.shape[-1]
    # print(f"{vocab_size=}")

    # # p.gather(dim=-1, index=x_t[..., None]) has the same shape as `index` which is (batch_size, seq_len, 1) in this case,
    # # each value is basically p_{t|0}(x_t^{i}|x_0^{i}). 
    # ratio = p / p.gather(dim=-1, index=x_t[..., None])  # (batch_size, seq_len, vocab_size)
    # assert not torch.isnan(ratio).any(), "DROSE"
    # assert not torch.isinf(ratio).any(), "OUU"
    # print(f"{ratio=}")
    # loss = (log_scores.exp() - ratio * log_scores).sum(dim=-1) / vocab_size  # (batch_size, seq_len) -- log_scores were zero'd when x_t^{i} = y in the forward pass, so the sum is correct here
    # print(f"[y]: {loss=}")
    # loss = (sigma[:, None] * loss).sum(dim=-1)  # (batch_size,)
    # print(f"[z]: {loss=}")
    # loss = loss.mean()  # scalar
    # return loss

"""
Now for the actual training loop...

We'll sample x_0 ~ p_data by having our dataloader give us batches of sequences of token ids from our dataset. It'll have shape (batch_size, seq_len)
We'll sample a batch of times t ~ Unif([0,1]), it'll have shape (batch_size,)

We can directly feed these into our loss function along with our model which outputs log scores and our CTMC that defines our forward noising process.
Since our loss averages across the entire batch, we can scale it for gradient accumulation easily. 

TODO inference, val loss, perplexity/gen perplexity+entropy, checkpointing, resuming training, sharded optimizer, fp16+scaler/bf16 handling, etc

"""

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def write0(s, log_file):
    """
    `s` is the string to write to the log file
    `log_file` is the path to the log.txt file
    """
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        with open(log_file, 'a') as f:
            f.write(s)

def create_log_dir(parent_dir):
    ddp_rank = int(os.environ.get('RANK', 0))
    log_dir = None # initialize as None to avoid errors on nonzero ranks
    if ddp_rank == 0:
        timestamp = datetime.now().strftime("%m-%d-%Y-%Hh%Mm%Ss")
        log_dir = os.path.join(parent_dir, f"{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, "checkpoints") # store checkpoints here
        os.makedirs(checkpoint_dir, exist_ok=True)
        sample_dir = os.path.join(log_dir, "samples") # store image samples made during training here
        os.makedirs(sample_dir, exist_ok=True)
    return log_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    effective_batch_size = config["effective_batch_size"]
    batch_size = config["batch_size"]
    grad_accum_steps = config["grad_accum_steps"]
    training_steps = config["training_steps"]

    val_loss_interval = config["val_loss_interval"]
    checkpoint_interval = config["checkpoint_interval"]
    text_sample_interval = config["text_sample_interval"]

    save_dir = config["save_dir"]

    log_dir = create_log_dir(save_dir)

    # Learning Rate Schedule (Cosine Decay)
    warmup_steps = config["warmup_steps"]
    max_lr = config["max_lr"]
    min_lr = config.get("min_lr", max_lr/10)
    lr_decay_steps = training_steps - warmup_steps
    def get_lr(it):
        # 1) linear warmup for warmup_steps steps
        if it < warmup_steps:
            return max_lr * (it + 1) / (warmup_steps + 1)
        # 2) if it > lr_decay_steps, return min learning rate
        if it > lr_decay_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)

    model_config = DiTConfig.from_dict(config["model"])
    model = DiT(model_config) # this model outputs the log of the scores

    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()

        assert rank == dist.get_rank(), f"{rank=}, {dist.get_rank()=}"

        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")

    # sanity check inputs
    if world_size * batch_size * grad_accum_steps != effective_batch_size:
        raise ValueError(f"{effective_batch_size=}, {world_size=}, {batch_size=}, {grad_accum_steps=}, {world_size*batch_size*grad_accum_steps=}")

    # Initialize log file
    log_file = f"{log_dir}/log.txt"
    if rank == 0:
        with open(log_file, 'w') as f:
            f.write("") # initialize log file

        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f) # save copy of yaml in log directory

    # Write basic info at start of log file (config, GPU info, etc.)
    write0("Config:\n", log_file=log_file)
    for key, value in config.items():
        if isinstance(value, dict):
            write0(f"  {key}:\n", log_file=log_file)
            for subkey, subvalue in value.items():
                write0(f"    {subkey}: {subvalue}\n", log_file=log_file)
        else:
            write0(f"  {key}: {value}\n", log_file=log_file)
    write0(f"Using {world_size} GPU(s)\n", log_file=log_file)
    write0(f"GPU Type: {torch.cuda.get_device_name()}\n", log_file=log_file)

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write0(f"Model Parameters: {num_params:,}\nTrainable Model Parameters: {num_trainable_params:,}\n", log_file=log_file)

    train_loader = ShardDataLoader(shard_dir=config["train_shard_dir"], batch_size=batch_size, seq_len=config["seq_len"])
    
    torch.manual_seed(config["rng_seed"] + rank) # for sampling times

    ctmc = UniformCTMC(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=config["AdamW_weight_decay"],
        betas=config["AdamW_betas"],
        eps=config["AdamW_epsilon"],
        fused=True
        )

    for step in range(training_steps):

        torch.cuda.synchronize()
        t0 = time.time()

        x0, t = train_loader.next_batch()
        x0 = x0.to(device)
        t = t.to(device)

        train_loss = 0.0 # for logging
        for micro_step in range(grad_accum_steps):

            if ddp: # only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # Effective batch size is per_gpu_batch_size * num_gpus * grad_accum_steps. Before syncing gradients, the local gradient has
            # per_gpu_batch_size in the denominator (since the loss is averaged over the local batch) and then when we sync gradients with
            # the .backward() call (with require_backward_grad_sync True), they are averaged over all ranks, so the resulting gradient has
            # per_gpu_batch_size * num_gpus in the denominator. Dividing the loss by grad_accum_steps gives us the correct final denominator.
            loss = loss_DWDSE(log_score_model=model, x_0=x0, t=t, ctmc=ctmc) / grad_accum_steps
            train_loss += loss.detach() # for logging
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"    [{micro_step=}] {name=}: gradient contains NaN")
                        print(f"    {param.grad=}")

        if dist.is_initialized():
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # for logging
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.time()

        write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss.item():.6f}    grad norm: {norm.item():.6f}\n", log_file=log_file)
    
    if ddp:
        dist.destroy_process_group()
