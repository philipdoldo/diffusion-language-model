import torch
import argparse
import yaml
import os
import math
import time
import torch.distributed as dist
import torch.nn.functional as F
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
at sequence position i) and we'll care about only the corresponding column of our transition matrix given by p_{t|0}(.|x_0^{i}). Notice
how when sigma_bar(t) gets big, the transition probabilities converge to the uniform distribution:
     exp(sigma_bar(t)*Q) = (1 - exp(-sigma_bar(t)))*(1/N)*11^T + exp(-sigma_bar(t))*I --> (1/N)*11^T as sigma_bar(t) --> +inf
This justifies the choice of sigma_min=1e-4 and sigma_max=20 as exp(-1e-4) ~= 0.999900005 and exp(-20) ~= 2.06115362e-9, so indeed the KL div
in the EBLO with L_DWDSE term will be really small
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
    
    def loss_DWDSE(self, log_score_model, x_0, t):
        """
        The Diffusion Weighted Denoising Score Entropy loss L_{DWDSE} from the SEDD paper https://arxiv.org/pdf/2310.16834 (see Algorithm 1)

        `log_score_model` is our neural network that outputs `log_scores`
        `x_0` (batch_size, seq_len)
        `t` (batch_size,)

        `x_t` (batch_size, seq_len) -- we use x_0 to get the transition probabilities p_{t|0}(.|x_0) so we can sample x_t
        `log_scores` (batch_size, seq_len, vocab_size) -- once we have x_t, we can use x_t and t to call our model to get the log scores
        
        We split the loss up into 3 terms: pos_term, neg_term, constant (all of which will be weighted by sigma(t) at the end), so we'll get
            sigma(t) * (pos_term - neg_term + constant)
        The gradient does not depend on the constant term, but without it our loss can be negative and harder to interpret

        pos_term is the sum of the scores s_{theta}(x_t, t)_{i, y} over all y=/=x_t^i for a given sequence position i. Shape: (batch_size, seq_len)
        neg_term is the p_{t|0}(y|x_0^i)/p_{t|0}(x_t^i|x_0^i) * log(s_{theta}(x_t, t)_{i, y}) summed over all y=/=x_t^i for a given sequence
        position i, where each term in the sum is a ratio multiplied by a log score. We can efficiently compute the ratios by considering a few cases:
        (N refers to the vocab size.)

            Case 1: x_t^i = x_0^i, y =/= x_t^i 
                ratio = p_{t|0}(y|x_0^i)/p_{t|0}(x_t^i|x_0^i) = p_{t|0}(y|x_0^i)/p_{t|0}(x_0^i|x_0^i) = 1 - N/(exp(sigma_bar(t)) - 1 + N)

            Case 2: x_t^i =/= x_0^i, y =/= x_t^i
                Case 2a: y =/= x_0^i
                    ratio = 1    (because neither y nor x_t^i are equal to x_0, they both have the same probability)
                Case 2b: y = x_0^i
                    ratio = 1 + N/(exp(sigma_bar(t)) - 1)    (this is just the reciprocal of the ratio from Case 1)

                    (note that Case 2b only matters for a single term in our sum over all vocabulary tokens y =/= x_t^t, it is the term corresponding
                    to the single time that y is equal to x_0^i, Case 2a will apply to all other terms in the sum)

        I got this implementation idea from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/graph_lib.py#L162
 
        """
        p = self.transition(cols=x_0, t=t) # (batch_size, seq_len, vocab_size)
        x_t = sample_categorical(p) # (batch_size, seq_len)
        seq_len = x_t.shape[-1]
        log_scores = log_score_model(token_ids=x_t, t=t) # (batch_size, seq_len, vocab_size)
        # TODO x_0 is unit16, but x_t is int64, why?

        scores = log_scores.exp() # (batch_size, seq_len, vocab_size)
        pos_term = scores.mean(dim=-1) - torch.gather(scores, -1, x_t[..., None]).squeeze(-1) / self.N # for each sequence position i, we subtract away the term in the scores where y == x_t^i (and we normalize all terms by N)

        sigma_bar = self.noise.sigma_bar(t[:, None]) # (batch_size, 1)
        # Compute  exp(sigma_bar(t)) - 1  in a numerically stable way
        em1 = torch.where(
            sigma_bar < 0.5, # SEDD code uses 0.5, seems to be an arbitrary threshold for numerical stability (I expect it could be a lot smaller than 0.5?)
            torch.expm1(sigma_bar),
            torch.exp(sigma_bar) - 1 # I guess it is computationally fast to do this when numerical stability isn't a concern
        ) # shape (batch_size, 1)

        # ratio = 1 - self.N / (em1 + self.N) # From Case 1: x_t^i == x_0^i, y != x_t^i, shape (batch_size, 1)
        ratio = em1 / (em1 + self.N) # Turns out I actually need to do this for better numerical stability!! No idea why original SEDD code didn't do this because I copied their parameters... fp32 floating point numbers are way more dense near zero than they are near 1, which is why this works

        # for a given batch index and sequence position i, this is the sum of log scores over all y in the vocabulary minus the one term where y == x_t^i (all normalized by N)
        neg_term = log_scores.mean(dim=-1) - torch.gather(log_scores, -1, x_t[..., None]).squeeze(-1) / self.N # (batch_size, seq_len)
        neg_term = torch.where(
            x_t == x_0.long(), # TODO
            ratio * neg_term, # Case 1: x_t^i == x_0^i, y != x_t^i 
            neg_term + torch.gather(log_scores, -1, x_0[..., None].long()).squeeze(-1) / em1 # Case 2a is ratio=1, Case 2b is ratio=1+N/em1, neg_term is already (1/N) multiplied by each log score, we just need to add 1/em1 multiplied by the log score for x_0^i to get the proper ratio for that term (remember, all terms are normalized by N)
        ) # shape (batch_size, seq_len)                         # TODO .long() here too

        # Constant term is sum of K(ratio) over all the ratios we used in neg_term, where K(a) := a*(log(a) - 1) 
        constant = torch.where(
            x_t == x_0.long(), # TODO
            ratio*(ratio.log() - 1) * (self.N - 1)/self.N, # Case 1, all ratios are the same and normalized by N, but we have N-1 terms, so (N-1)/N times the same K(ratio)
            ((-ratio.log() - 1) / ratio - (self.N - 2)) / self.N  # for all but one term, we get K(1) = 1*(log(1) - 1) = -1 (there are N-2 of these terms), for the remaining term the ratio we want is the reciprocal of the Case 1 ratio, so we use 1/ratio and -log(ratio) to get K(1/ratio) = (-ratio.log() - 1) / ratio. Finally, we normalize everything by N
        ) # (batch_size, seq_len)

        sigma = self.noise.sigma(t) # (batch_size,)
        loss = sigma * (pos_term - neg_term + constant).sum(dim=-1) / seq_len # (batch_size,) (I normalize by sequence length too even though I don't see that in their code)

        return loss.mean() # scalar

    def rate(self, cols, t):
        """
        `t` shape (batch_size,)
        `cols` shape (batch_size, seq_len) -- these will be sequences of token ids
        Returns columns of the rate matrix Q_t = sigma(t) * Q
        """
        batch_size, seq_len = cols.shape
        Q = torch.ones((batch_size, seq_len, self.N), device=cols.device) / self.N # all non-diagonal entries of Q are 1/N
        Q.scatter_(-1, cols[..., None], (1-self.N)/self.N) # diagonal entries of Q are (1-N)/N 
        sigma = self.noise.sigma(t) # (batch_size,)
        Q_t = sigma[:, None, None] * Q # (batch_size, seq_len, vocab_size)
        return Q_t

    def reverse_rate(self, scores, cols, t):
        """
        `scores` has shape (batch_size, seq_len, vocab_size) -- be sure these are the scores and not the log scores
        `cols` shape (batch_size, seq_len) -- these will be sequences of token ids
        `t` shape (batch_size,)
        Returns columns of the reverse rate matrix. 

        Recall that the reverse rate matrix satisfies Q_t^{reverse}(y|x) = Q_t(x|y)*(p_t(y)/p_t(x)) for y != x. So we'd need to transpose the forward
        rate matrix, but for uniform diffusion it is symmetric so we don't need to bother transposing. Also, note that the factor of p_t(y)/p_t(x) will
        be replaced by our learned scores.
        """
        Q_t = self.rate(cols, t) # no need to transpose for uniform diffusion    (batch_size, seq_len, vocab_size)
        reverse_Q_t = Q_t * scores # element-wise multiplication, we need to correct the diagonal entries though    (batch_size, seq_len, vocab_size)
        reverse_Q_t.scatter_(-1, cols[..., None], torch.zeros_like(reverse_Q_t)) # set diagonals to zero so we don't count them in our sum
        reverse_Q_t.scatter_(-1, cols[..., None], -reverse_Q_t.sum(dim=-1, keepdim=True)) # diagonals are negative sum of nondiagonal terms
        return reverse_Q_t # (batch_size, seq_len, vocab_size)

    def forward_euler_sample(self, log_score_model, batch_size, seq_len, num_steps, device):
        """
        The most naive sampling approach: use forward Euler on the reverse process from t=1 to t=0

        Note that we can linearize p_{t+h|t}(.|x) ~= p_{t|t}(.|x) + h * Q_t(.|x) = I(.|x) + h * Q_t(.|x) (where I(.|x) denotes column x of the identity matrix).
        If h is sufficiently small, then this will be a probability distribution, but in practice h might not be small enough and so we'll need to clamp values
        into [0, 1] and normalize it into a probability distribution. Also, we will be doing this for the reverse process, so we want to approximately sample 
        from p_{t-h|t}(.|x_t) at each step.
        """
        if not (isinstance(num_steps, int) and num_steps > 0):
            raise ValueError(f"{num_steps=}, but it must be a positive integer")
        step_size = 1/num_steps

        # sample initial noise from the uniform distribution (random token ids) at time t=1
        t = torch.ones(batch_size, device=device) # (batch_size,)
        x_t = torch.randint(low=0, high=self.N, size=(batch_size, seq_len), device=device) # (batch_size, seq_len)
        for _ in range(num_steps):

            log_scores = log_score_model(token_ids=x_t, t=t) # (batch_size, seq_len, vocab_size)
            scores = log_scores.exp()

            reverse_Q_t = self.reverse_rate(scores=scores, cols=x_t, t=t)

            # Approximate p_{t-h|t}(.|x_t), clamp and normalize because it might not be a probability distribution otherwise
            p = F.one_hot(x_t, num_classes=self.N) + step_size * reverse_Q_t # (batch_size, seq_len, vocab_size) -- may need to clamp and normalize along vocab dimension to ensure we get valid probability distributions
            p = p.clamp(min=0.0, max=1.0)
            p = p / p.sum(dim=-1, keepdim=True)

            t = t - step_size
            x_t = sample_categorical(p)
        return x_t
    

def sample_categorical(p):
    """
    `p` has shape (batch_size, seq_len, vocab_size) and is basically p_{t|0}(.|x_0^{i}) for each sequence index i and each batch index when training.
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


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters, i.e., ema_params = (1-a)*new_params + a*ema_params for a in [0,1]
    Based on: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/ema.py#L10

    When training diffusion models, people often use an EMA of weights for inference instead of the actual weights used during training
    """
    def __init__(self, params, decay=0.9999):
        """
            `params`: Iterable of `torch.nn.Parameter`; usually the result of `model.parameters()`.
            `decay` : float in [0,1]
        """
        if decay < 0 or decay > 1:
            raise ValueError(f"Decay must be in [0,1], but {decay=}")
        self.decay = decay
        self.ema_params = [p.clone().detach() for p in params if p.requires_grad]
        self.copied_params = []

    def update(self, params):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of the `optimizer.step()` call.
        Args:
            params: Iterable of `torch.nn.Parameter`; usually the same set of parameters used to initialize this object.
        """
        with torch.no_grad():
            for ema_p, p in zip(self.ema_params, [p for p in params if p.requires_grad]):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay) # ema_p = decay*ema_p + (1-decay)*p, update ema params in-place

    def copy_to(self, params):
        """
        Copy EMA parameters into given collection of parameters.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored moving averages.
        """
        for ema_p, p in zip(self.ema_params, [p for p in params if p.requires_grad]):
            p.data.copy_(ema_p.data)

    def store(self, params):
        """
        Save the current parameters for restoring later.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be temporarily stored.
        """
        self.copied_params = [p.clone() for p in params]

    def restore(self, params):
        """
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the `copy_to` method. After validation (or model saving), use this
        to restore the former parameters.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored parameters.
        """
        for t, p in zip(self.copied_params, params):
            p.data.copy_(t.data)

    def state_dict(self):
        return dict(decay=self.decay, ema_params=self.ema_params)

    def load_state_dict(self, state_dict, device=None):
        self.decay = state_dict['decay']
        self.ema_params = state_dict['ema_params']
        if device is not None:
            self.ema_params = [ema_p.to(device) for ema_p in self.ema_params]

"""
Now for the actual training loop...

We'll sample x_0 ~ p_data by having our dataloader give us batches of sequences of token ids from our dataset. It'll have shape (batch_size, seq_len)
We'll sample a batch of times t ~ Unif([0,1]), it'll have shape (batch_size,)

We can directly feed these into our loss function along with our model which outputs log scores and our CTMC that defines our forward noising process.
Since our loss averages across the entire batch, we can scale it for gradient accumulation easily. 

TODO inference, parallelize val loss, perplexity/gen perplexity+entropy, resuming training, sharded optimizer, fp16+scaler/bf16 handling, etc

"""

def print0(s="", **kwargs):
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(s, **kwargs)

def write0(s, log_file):
    """
    `s` is the string to write to the log file
    `log_file` is the path to the log.txt file
    """
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        with open(log_file, 'a') as f:
            f.write(s)

def create_log_dir(parent_dir):
    rank = int(os.environ.get('RANK', 0))
    log_dir = None # initialize as None to avoid errors on nonzero ranks
    if rank == 0:
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

    # Learning Rate Schedule (Cosine Decay -- warmup + constant if you let min_lr = max_lr and cosine_decay_steps=0)
    warmup_steps = config["warmup_steps"]
    max_lr = config["max_lr"]
    min_lr = config.get("min_lr", max_lr/10)
    lr_decay_steps = config.get("cosine_decay_steps", training_steps - warmup_steps)
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

    if config.get("resume_training", False):
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
        # If we resume training, we change the rng seed in the config as a lazy way making sure we don't get the same random times and such
        # I didn't bother storing rng state of each rank because I might resume with a different number of gpus anyway and it is simpler this way
        # The checkpoint stores a set of all rng seeds used across all training runs to be sure we never repeat any of them (the gpus I'm using can have a lot of issues)
        if config["rng_seed"] in checkpoint["rng_seeds"]:
            raise ValueError(f"Change the rng seed in the config before you resume training! {checkpoint['rng_seeds']=}, {config['rng_seed']=}")

        model.load_state_dict(checkpoint["model"])
        print0(f"MODEL LOADED WITH CHECKPOINT {config['checkpoint_path']}\n")
    prior_rng_seeds = checkpoint["rng_seeds"] if config.get("resume_training", False) else set() # to be stored in checkpoint to be sure we don't accidentally resume training with a previously used rng seed
    prior_rng_seeds.add(config["rng_seed"])

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

    train_loader = ShardDataLoader(shard_dir=config["train_shard_dir"], batch_size=batch_size, seq_len=config["seq_len"], rng_seed=config["rng_seed"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=config["AdamW_weight_decay"],
        betas=config["AdamW_betas"],
        eps=config["AdamW_epsilon"],
        fused=True
        )
    
    ema = ExponentialMovingAverage(params=model.parameters(), decay=config["ema_decay"])

    if config.get("resume_training", False):
        train_loader.load_state_dict(checkpoint["dataloader"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"], device=device)

        initial_step = checkpoint["step"]

        write0(f"RESUMING TRAINING WITH CHECKPOINT {config['checkpoint_path']} AT STEP {initial_step}\n", log_file=log_file)
    else:
        initial_step = 0 # if not resuming training, have training loop start at step 0

    torch.manual_seed(config["rng_seed"] + rank) # (I guess this affects the categorical sampling, random times are in the dataloader)
    ctmc = UniformCTMC(config)

    for step in range(initial_step, training_steps):

        # SAVE CHECKPOINTS
        if rank == 0 and (step % checkpoint_interval == 0 or step == training_steps - 1):
            torch.cuda.synchronize()
            t0 = time.time()
            checkpoint = {
                'step' : step,
                'model' : model.module.state_dict() if ddp else model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'dataloader' : train_loader.get_state_dict(),
                'ema' : {'ema_params' : ema.ema_params, 'decay' : ema.decay},
                'rng_seeds' : prior_rng_seeds,
            }
            checkpoint_path = os.path.join(log_dir, f'checkpoints/checkpoint_step{step}.pt')
            torch.save(checkpoint, checkpoint_path)
            torch.cuda.synchronize()
            t1 = time.time()
            write0(f" --- Checkpoint saved to {checkpoint_path} in {t1-t0:.4f}s\n", log_file=log_file)

        if rank == 0 and (step % val_loss_interval == 0 or step == training_steps - 1):
            with torch.no_grad():
                
                t0 = time.time()
                rng_state = torch.get_rng_state() # val might change rng state on rank 0, so save and restore it just in case, probably not very important
                val_loader = ShardDataLoader(shard_dir=config["val_shard_dir"], batch_size=batch_size, seq_len=config["seq_len"]) # Should be reinitialized with same rng seed every time. Also notice how I intentionally use the default rng seed for val loader so that it never changes even when I resume training with a new rng seed in my config
                val_loader.reset() # should be unnecessary

                ema.store(model.parameters()) # store copy of the actual model weights
                ema.copy_to(model.parameters()) # copy EMA weights into the model
                val_losses = []
                for val_step in range(config.get("val_steps", 98)):
                    val_x0, val_t = val_loader.next_batch()
                    val_x0 = val_x0.to(device)
                    val_t = val_t.to(device)
                    val_loss = ctmc.loss_DWDSE(log_score_model=model, x_0=val_x0, t=val_t)
                    val_losses.append(val_loss)
                val_loss = sum(val_losses) / len(val_losses)
                ema.restore(model.parameters()) # copy stored model weights back into the model
                torch.set_rng_state(rng_state) # restore rng state on rank 0
                t1 = time.time()
                write0(f"val loss: {val_loss}{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms\n", log_file=log_file)

        torch.cuda.synchronize()
        t0 = time.time()

        train_loss = 0.0 # for logging
        for micro_step in range(grad_accum_steps):

            x0, t = train_loader.next_batch()
            x0 = x0.to(device)
            t = t.to(device)

            if ddp: # only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # Effective batch size is per_gpu_batch_size * num_gpus * grad_accum_steps. Before syncing gradients, the local gradient has
            # per_gpu_batch_size in the denominator (since the loss is averaged over the local batch) and then when we sync gradients with
            # the .backward() call (with require_backward_grad_sync True), they are averaged over all ranks, so the resulting gradient has
            # per_gpu_batch_size * num_gpus in the denominator. Dividing the loss by grad_accum_steps gives us the correct final denominator.
            loss = ctmc.loss_DWDSE(log_score_model=model, x_0=x0, t=t) / grad_accum_steps
            train_loss += loss.detach() # for logging
            loss.backward()

        if dist.is_initialized():
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # for logging
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)
        optimizer.step()
        ema.update(model.parameters())
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.time()

        write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss.item():.6f}    grad norm: {norm.item():.6f}\n", log_file=log_file)
    
    if ddp:
        dist.destroy_process_group()
