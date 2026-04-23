# diffusion-language-model
I implemented a diffusion language model for my own learning experience based on the [SEDD paper](https://arxiv.org/abs/2310.16834). 

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

# Background
In continuous flow/diffusion models, we work with ODEs/SDEs (with initial conditions sampled according to some initial probability distribution) which result in Markov processes. In discrete diffusion models we work with Markov chains. In particular, we will work with continuous-time Markov chains (CTMCs). I'll provide some notes on CTMCs below, but it is not at all intended to be fully rigorous or comprehensive. 
## Continuous-Time Markov Chains
We denote a CTMC by the stochastic process $\\{ x_t \\}_{t \geq 0}$ indexed by time. At each time $t$ the process $x_t$ belongs to a single state in the *state space* $S := \\{1, ..., m \\}$ (we are assuming a finite state space). 

**Remark:** In the context of diffusion language models, suppose our vocabulary is a set of tokens $V$. In our case, using the GPT-2 tokenizer we have |V| = 50257. Suppose further that we training a model to generate a sequence of $d$ tokens. In this case, our state space consists of all length- $d$ sequences of tokens in $V$ so that $m=|V|^d$. For even moderate sequence lengths (e.g., I used $d=1024$), this is an intractably large number to deal with as our rate and transition matrices are $m \\times m$. In practice, we introduce a sparsity constraint where a sequence of tokens can only transition to a sequence of tokens that differs from the original sequence by at most one token -- this will ultimately make things tractable and each individual sequence position can effectively use a $|V| \\times |V|$ rate matrix (see Section 3.3 of the [SEDD paper](https://arxiv.org/abs/2310.16834) for more details).

A CTMC will transition to other states as determined by time-dependent transition probabilities. For times $s \\leq t$ let $p_{t|s}$ be the column-stochastic $m \\times m$ transition matrix which is indexed by the CTMC's states so that for $x, y \\in S$ we have 

$$p_{t|s}(x|y) := P(x_t = x | x_s = y)$$

which denotes the probability that the process will be in state $x$ at time $t$ given that the process is in state $y$ at time $s$ and is given by row $x$ and column $y$ of $p_{t|s}$.

**Remark** In the classical CTMC literature, it is often assumed that the transition probabilities are row-stochastic. This is just a convention. In much of the diffusion language model literature, a column-stochastic convention is used. Additionally, the classical CTMC literature tends to make an additional assumption that the CTMC is time-homogeneous, which means that $P(x_t = x | x_s = y) = P(x_{t - s} = x | x_0 = y)$ so that all that matters is how much time has passed since transitioning to the current state and thus the transition probabilities only need to depend on a single time argument under this assumption of time homogeneity. It is also worth noting that homogeneous CTMCs have constant rate matrices, but nonhomogeneous CTMCs have time-varying rate matrices. We do not assume time homogeneity in the context of diffusion language models, our rate matrices will be time-varying and our transition probabilities will depend on two time arguments. 


