# diffusion-language-model
I implemented a diffusion language model for my own learning experience. I based it on the [SEDD paper](https://arxiv.org/abs/2310.16834). 

I wanted to keep the code very simple with very few abstractions. I chose to only implement uniform diffusion to keep the code short (and once you understand how it works, adding masked diffusion should be trivial). Even though masked diffusion has traditionally gotten better results than uniform diffusion in the literature, I personally like how uniform diffusion has the ability to edit an individual token multiple times whereas with masked diffusion once a token is unmasked it remains committed to that same token (at least in the theoretical reverse process, which only approximately gets learned in practice).

Below I describe the code structure and give a little theoretical background.
# Code Structure
- `dataloader.py` implements a basic dataloader that handles data paralleism for multi-gpu/multi-node training. I would probably use memmaps instead of numpy shards if I did it again.
- `model.py` implements a [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748) architecture using bidirectional attention, adaLN-Zero, RoPE, etc. I likely made some nonstandard choices (e.g., using RMS norm without learnable parameters in some places), I did not intend to faithfully replicate existing archtectures perfectly.
- `sample.py` is a short test script to try unconditional sampling from a checkpoint. I currently only have forward Euler sampling implemented (since that was all I needed for a basic proof of concept), maybe I'll improve that later.
- `template.yaml` is an example config for running the training script.
- `tokenize_owt.py` downloads OpenWebText (OWT) and tokenizes it using the GPT-2 tokenizer. It ends up being ~9B tokens. I reserved the last 100K tokens for validation. I stored the tokenized data in numpy array shards, but if I redid it I would probably use memmaps instead.
- `train.py` is the training script which also contains some classes in it that I could have put in other files (e.g., `ExponentialMovingAverage`, `GeometricNoise`, `UniformCTMC`). Maybe I'll reorganize this, but I think the code is pretty simple as it is. I wrote a lot of comments to provide theoretical justification for various parts of the code. I might cover them later on in this readme.
- `train.sh` is just a slurm script I used and committed for my own convenience.

There are some notable omissions from the code. I won't list them all, but here's one: I actually did not used mixed precision during training specifically because I only had access to some V100s (yes, in the year 2026) which don't support bf16 and instead support fp16. I chose to do my first run of training entirely in fp32 because there are some places where numerical stability matters and some literature suggests that normalization layers should be done in fp32. Things probably work with fp16, but I just wanted to play it safe before introducing other potential sources of error. If I do add mixed precison, I'll need to check whether or not the normalization layers need fp32 or if they can be fp16 as well. 

# Some Notes
In continuous flow/diffusion models, we work with ODEs/SDEs (with initial conditions sampled according to some initial probability distribution) which result in Markov processes. In discrete diffusion models we work with Markov chains. In particular, we will work with continuous-time Markov chains (CTMCs). I'll provide some notes on CTMCs below, but they are not at all intended to be fully rigorous or comprehensive. The notation used below might make some of the comments in my code easier to understand.
## Continuous-Time Markov Chains
We denote a CTMC by the stochastic process $\\{ x_t \\}_{t \geq 0}$ indexed by time. At each time $t$ the process $x_t$ belongs to a single state in the *state space* $S := \\{1, ..., m \\}$ (we are assuming a finite state space). 

**Remark:** In the context of diffusion language models, suppose our vocabulary is a set of tokens $V$. In our case, using the GPT-2 tokenizer we have |V| = 50257. Suppose further that we training a model to generate a sequence of $d$ tokens. In this case, our state space consists of all length- $d$ sequences of tokens in $V$ so that $m=|V|^d$. For even moderate sequence lengths (e.g., I used $d=1024$), this is an intractably large number to deal with as our rate and transition matrices are $m \\times m$. In practice, we introduce a sparsity constraint where a sequence of tokens can only transition to a sequence of tokens that differs from the original sequence by at most one token -- this will ultimately make things tractable and each individual sequence position can effectively use a $|V| \\times |V|$ rate matrix (see Section 3.3 of the [SEDD paper](https://arxiv.org/abs/2310.16834) for more details).

A CTMC will transition to other states as determined by time-dependent transition probabilities. For times $s \\leq t$ let $p_{t|s}$ be the column-stochastic $m \\times m$ transition matrix which is indexed by the CTMC's states so that for $x, y \\in S$ we have 

$$p_{t|s}(x|y) := P(x_t = x | x_s = y)$$

which denotes the probability that the process will be in state $x$ at time $t$ given that the process is in state $y$ at time $s$ and is given by row $x$ and column $y$ of $p_{t|s}$.

**Remark:** In the classical CTMC literature, it is often assumed that the transition probabilities are row-stochastic. This is just a convention. In much of the diffusion language model literature, a column-stochastic convention is used. Additionally, the classical CTMC literature tends to make an additional assumption that the CTMC is time-homogeneous, which means that $P(x_t = x | x_s = y) = P(x_{t - s} = x | x_0 = y)$ so that all that matters is how much time has passed since transitioning to the current state and thus the transition probabilities only need to depend on a single time argument under this assumption of time homogeneity. It is also worth noting that homogeneous CTMCs have constant rate matrices, but nonhomogeneous CTMCs have time-varying rate matrices. We do not assume time homogeneity in the context of diffusion language models, our rate matrices will be time-varying and our transition probabilities will depend on two time arguments. 

We can define the (time-dependent) **rate matrix** 

$$Q_t := \lim_{h \to 0^+} \frac{p_{t+h|t} - p_{t|t}}{h} = \lim_{h \to 0^+} \frac{p_{t+h|t} - I}{h}$$

where $I$ is the $m \\times m$ identity matrix. We have that $p_{t|t} = I$ because intuitively if no time has passed, there should be zero probability of transitioning to a new state and the process should remain in its current state with probability 1. Based on this definition, it is easy to show that: 

1. the columns of $Q_t$ sum to 0
2. the nondiagonal entries of $Q_t$ are nonnegative
3. the diagonal entries of $Q_t$ are the negative sum of all nondiagonal entries in the corresponding column

In practice, we will define a CTMC using its rate matrix rather than its more complicated transition probabilities. We can solve an ODE (the Kolmogorov Forward Equation) to get the transition probabilities. 

We can show that for times $s < u < t$, the transition probabilities satisfy the **Chapman-Kolmogorov equation**:

$$p_{t|s} = p_{t|u} p_{u|s}$$

which can be shown as follows (making use of the Markov property):

$$p_{t|s}(x|y) = P(x_t=x|x_s=y) = \sum_{z \in S}P(x_t=x,x_u=z|x_s=y) = \sum_{z \in S} P(x_t=x| x_u=z, x_s=y)P(x_u=z|x_s=y) = \sum_{z \in S} P(x_t=x | x_u=z)P(x_u=z|x_s=y) = \sum_{z \in S} p_{t|u}(x|z)p_{u|s}(z|y)$$

We can use the Chapman-Kolmogorov equation to get the **Kolmogorov Forward Equation (KFE)**:

$$\frac{\partial}{\partial t} p_{t|s} = Q_t p_{t|s}$$

$$\frac{\partial}{\partial t} p_{t|s} = \lim_{h \to 0^+} \frac{p_{t+h|s} - p_{t|s}}{h} = \lim_{h \to 0^+} \frac{p_{t+h|t} p_{t|s} - p_{t|s}}{h} = \lim_{h \to 0^+} \frac{p_{t+h|t} - I_{m \times m}}{h} p_{t|s} = Q_t p_{t|s}$$

The KFE also applies to marginals:

$$\frac{d}{dt} p_t = Q_t p_t$$

$$\frac{d}{dt} p_t = \frac{d}{dt} p_{t|0}p_0 = Q_t p_{t|0} p_0 = Q_t p_t$$ 

(We used the fact that $p_{t|0}$ solves the KFE.)

We know that the marginal satisfies this ODE, but under some regularity conditions it can be shown that it is the unique solution for some initial condition given by $p_0$ by existence and uniqueness for ODEs. We can view the KFE as a discrete analog of the Fokker-Planck equation (continuity equation) used in continuous diffusion (flow) models. 

In the time-homogeneous case, the rate matrix $Q$ is not time dependent and we can solve the KFE to get

$$p_t = exp(Qt).$$

In a slightly more complicated (but much more practical) case, we can assume a simple form for the rate matrix $Q_t := \sigma(t) Q$ for some scalar function $\sigma$ (this is what is done in the SEDD paper and my code) in which case solving the KFE gives

$$p_{t|s} = \exp\left( \int_{s}^{t} \sigma(\tau) d \tau \, Q \right) := \exp\left( (\overline{\sigma}(t) - \overline{\sigma}(s)) \, Q \right) $$

where $\overline{\sigma}(t) := \int_{0}^{t} \sigma(\tau) d \tau$ (note that we used the initial condition $p_{s|s} = I$). Note that the marginals satisfy $p_t = \exp\left(\overline{\sigma}(t) Q \right) p_0$ which is effectively the solution we get for the time-homogeneous case except the time has been transformed from $t$ to $\overline{\sigma}(t)$ (in my code, this is the quantity that I refer to as `sigma_bar`). More generally, for a time-varying rate matrix $Q_t$, solving the KFE would result in a time-ordered exponential 

$$p_{t|s} = I + \sum_{k=1}^{\infty} \int_{s}^{t} \int_{s}^{\tau_1} \cdots \int_{s}^{\tau_k} Q_{\tau_1} \cdots Q_{\tau_k d \tau_1 \cdots d \tau_k}$$

which happens because in general $Q_t Q_s \neq Q_s Q_t$ (that is, the rate matrices at different times do not commute in general -- in fact, this is usually the case) You can obtain the time-ordered exponential by doing Picard iteration on the KFE. You can show that if all of the rate matrices commute, then it reduces to $p_{t|s} = \exp(\int_{s}^{t} Q_{\tau} d \tau)$. Thus, the assumption made in the SEDD paper that $Q_t = \sigma(t) Q$ simplifies the expression for the transition probabilities considerably. 

For training a language model, we use a CTMC to define our forward noising process from pure data to pure noise to corrupt data at varying degrees that the model can learn to denoise. However, during inference in language modeling, we'll want to go from pure noise to pure data, so we'll want to simulate the reverse process. If our forward process goes from time $t=0$ to $t=T$, then the reverse process is the forward process with time $t$ replaced by $T-t$. Technically, we'd want to be careful and make sure that the reverse process itself is actually Markov. This is actually easy to see if you consider that an equivalent way of characterizing a Markov process is as a process where the future and past are conditionally independent when given the present. The future and past may change roles when switching between the forward and reverse processes, but conditional independence w.r.t. the present state still holds and so the reverse process is Markov. As such, we can find a rate matrix $\overline{Q}_t$ for the reverse process CTMC.

A way to get some intuiton for the reverse rate matrix $\overline{Q}_t$ is by perturbing time in the reverse direction and using Bayes (for $x,y \in S$ with $x \neq y$):

$$\overline{Q}_t(y|x) := \lim_{h \to 0+} \frac{p_{t-h|t}(y|x) - p_{t|t}(y|x)}{h} = \lim_{h \to 0+} \frac{p_{t|t-h}(x|y) \frac{p_{t-h}(y)}{p_{t}(x)} - 0}{h} = Q_t(x|y) \frac{p_t(y)}{p_t(x)}$$

The diagonal entries (i.e., when $x = y$) are obtained by the fact that the nondiagonal entries are nonnegative and columns sum to zero.

The ratios $\frac{p_t(y)}{p_t(x)}$ are referred to as the *scores* or *concrete scores*, analogous to the *score* that shows up in a correction to the drift term when time reversing an SDE in continuous diffusion. Given the simple structure of our forward rate matrix $Q_t = \sigma(t) Q$, we only need to learn the scores to effectively simulate the reverse process. 
