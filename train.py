import torch
import argparse
import yaml

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
        return self.sigma_min**(1-t) * self.sigma_max**t * (torch.log(self.sigma_max) - torch.log(self.sigma_min))

"""
For uniform diffusion, we are using Q = (11^T - N*I)/N = (1/N)*11^T - I and (1/N)*11^T is a projection matrix, so the matrix
exponential is very easy to compute. This makes it so that 
    exp(sigma_bar(t)*Q) = (1 - exp(-sigma_bar(t)))*(1/N)*11^T + exp(-sigma_bar(t))*I
which defines our transition probabilities p_{t|0}. In practice, we are going to start at some initial state x_0^{i} (a given token
at sequence position i) and we'll care about only the corresponding column of our transition matrix given by p_{t|0}(.|x_0^{i}).
"""

class UniformCTMC:

    def __init__(self, config):
        self.noise = GeometricNoise(sigma_min=config.sigma_min, sigma_max=config.sigma_max)
        self.N = config.vocab_size # the rate and transition matrices are N-by-N (this is all at the token level rather than the sequence level)

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
        #             p[i][j][cols[i][j][k]] += c[i] # assuming c has shape (batch_size, seq_len, 1), which doesn't happen without .expand()
        p.scatter_add_(-1, cols[..., None], c.expand(batch_size, seq_len, 1))
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
    p = ctmc.transition(cols=x_0, t=t) # (batch_size, seq_len, vocab_size)
    x_t = sample_categorical(p) # (batch_size, seq_len)
    log_scores = log_score_model(token_ids=x_t, t=t) # (batch_size, seq_len, vocab_size)
    sigma = ctmc.noise.sigma(t) # (batch_size,)

    # p.gather(dim=-1, index=x_t[..., None]) has the same shape as `index` which is (batch_size, seq_len, 1) in this case,
    # each value is basically p_{t|0}(x_t^{i}|x_0^{i}). 
    ratio = p / p.gather(dim=-1, index=x_t[..., None])  # (batch_size, seq_len, vocab_size)
    loss = (log_scores.exp() - ratio * log_scores).sum(dim=-1)  # (batch_size, seq_len) -- log_scores were zero'd when x_t^{i} = y in the forward pass, so the sum is correct here
    loss = (sigma[:, None] * loss).sum(dim=-1)  # (batch_size,)
    loss = loss.mean()  # scalar
    return loss

"""
Now for the actual training loop...

We'll sample x_0 ~ p_data by having our dataloader give us batches of sequences of token ids from our dataset. It'll have shape (batch_size, seq_len)
We'll sample a batch of times t ~ Unif([0,1]), it'll have shape (batch_size,)

We can directly feed these into our loss function along with our model which outputs log scores and our CTMC that defines our forward noising process.
Since our loss averages across the entire batch, we can scale it for gradient accumulation easily. 

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    # TODO define dataloader and inference sampling

    
    torch.manual_seed(config.rng_seed + rank) # for sampling times


    