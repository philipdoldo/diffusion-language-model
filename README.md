# diffusion-language-model
I implemented a diffusion language model for my own learning experience based on the [SEDD paper](https://arxiv.org/abs/2310.16834). I wanted to keep the code very simple with very few abstractions. I chose to only implement uniform diffusion to keep the code short (and once you understand how it works, adding masked diffusion should be trivial). Even though masked diffusion has traditionally gotten better results than uniform diffusion in the literature, I personally like how uniform diffusion has the ability to edit an individual token multiple times whereas with masked diffusion once a token is unmasked it remains committed to that same token (at least in the theoretical reverse process, which only approximately gets learned in practice).
# Code Structure
- `dataloader.py` implements a basic dataloader that handles data paralleism for multi-gpu/multi-node training. I would probably use memmaps instead of numpy shards if I did it again.
- `model.py` implements a [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748) architecture using bidirectional attention, adaLN-Zero, RoPE, etc. I likely made some nonstandard choices (e.g., using rmsnorm without learnable parameters in some places), I did not intend to faithfully replicate existing archtectures perfectly.
- `template.yaml` is an example config for running the training script.
- `tokenize_owt.py` downloads OpenWebText (OWT) and tokenizes it using the GPT-2 tokenizer. I reserved the last 100K tokens for validation. I stored the tokenized data in numpy array shards, but if I redid it I would probably use memmaps instead.
- `train.py` is the training script which also contains some classes in it that I could have put in other files (e.g., `ExponentialMovingAverage`, `GeometricNoise`, `UniformCTMC`). Maybe I'll reorganize this, but I think the code is pretty simple as it is. I wrote a lot of comments to provide theoretical justification for various parts of the code, mostly as notes for myself. I might cover them later on in this readme.
- `train.sh` is just a slurm script I used and committed for my own convenience.

# Background
