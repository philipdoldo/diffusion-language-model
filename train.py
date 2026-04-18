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
        # TODO TODO TODO will `t` be a scalar or will it be shape (B,) or something?
        if t > 1 or t < 0:
            raise ValueError(f"Expected t in [0,1], got {t=}")
        return self.sigma_min**(1-t) * self.sigma_max**t
    
    def sigma(self, t):
        """
        This is the derivative of sigma_bar(t). In practice, we define sigma_bar(t) first and then take its derivative to get
        sigma(t) rather than starting with sigma(t) and computing sigma_bar(t) = int_{0}^{t} sigma(s) ds. There isn't a principled
        reason for this, this just happens to be what was done for this geometric noise schedule used in the SEDD paper.
        """
        # TODO raise ValueError()
        return sigma_min**(1-t) * sigma_max**t * (torch.log(sigma_max) - torch.log(sigma_min))

"""
For uniform diffusion, we are using Q = (11^T - N*I)/N = (1/N)*11^T - I and (1/N)*11^T is a projection matrix, so the matrix
exponential is very easy to compute. This makes it so that 
    exp(sigma_bar(t)*Q) = (1 - exp(-sigma_bar(t)))*(1/N)*11^T + exp(-sigma_bar(t))*I
which defines our transition probabilities p_{t|0}. In practice, we are going to start at some initial state x_0^{i} (a given token
at sequence position i) and we'll care about only the corresponding column of our transition matrix given by p_{t|0}(.|x_0^{i}).
"""
# TODO define transition probabilities and such