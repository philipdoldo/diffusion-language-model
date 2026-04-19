import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import yaml
import math

def rmsnorm(x):
    """
    `x` has shape (B, L, d) and the RMS (which is just 2-norm scaled by 1/sqrt(d) in R^d) is computed for every vector of channels
    No learnable parameters. I want to try rmsnorm since I used it in language models. 
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() # shape (B, L, 1)
    return (x / (rms + 1e-8))


def rmsnorm32(x):
    """
    rmsnorm but now with float32 precision just in case that ends up being important for stability
    """
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    return (x / (rms + 1e-8)).type(x.dtype)


class SinusoidalEmbedding(nn.Module):
    """Generates a conditioning embedding, an embedding conditioned on time in our case"""
    def __init__(self, cond_dim=256, base=10000):
        super().__init__()
        if cond_dim % 2 != 0:
            raise ValueError(f"Conditioning embedding dimension {cond_dim=} must be a multiple of 2")
        self.cond_dim = cond_dim # conditioning embedding dimension
        self.base = base # e.g. 10000
    
    def forward(self, t):
        """
        `t` has shape (B,) --- batch of times in [0, 1]
        `t` gets mapped to a batch of embeddings of dimension cond_dim --- output has shape (B, cond_dim)
        """
        i = torch.arange(self.cond_dim // 2, device=t.device) # shape (cond_dim//2)
        freqs = 1/self.base ** ((2 * i) / self.cond_dim) # shape (cond_dim//2)
        angles = freqs * t[:, None] # (cond_dim//2) * (B, 1) --> broadcasts to (B, cond_dim//2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Maps times in [0,1] to embedding vectors"""
    def __init__(self, input_dim, hidden_dim, output_dim, base):
        super().__init__()
        self.sine_emb = SinusoidalEmbedding(cond_dim=input_dim, base=base)
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, t):
        """
        `t` has shape (batch_size,)
        output has shape (batch_size, output_dim)
        """
        c = self.sine_emb(t)
        c = self.mlp(c)
        return c


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim
        output_dim = output_dim if output_dim is not None else input_dim

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.W2 = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        """
        `x` has shape (B, d) where B is batch size and d is embedding dimension
        """
        x = self.W1(x)
        x = F.silu(x) # TODO pass in different activation function options from config
        x = self.W2(x)
        return x
    

class AdaLNProjection(nn.Module):
    """
    generates the 6 scale/shift/gates from adaLN-Zero, see DiT paper https://arxiv.org/pdf/2212.09748 

    scales: gamma1, gamma2 in DiT paper
    shifts: beta1, beta2 in DiT paper
    gates: alpha1, alpha2 in DiT paper

    default behavior outputs a 6-tuple of tensors of shape (batch_size, 1, embed_dim)
    We need the 1 dimension to allow for broadcasting with tensors of shape (batch_size, seq_len, embed_dim) in the DiT
    """
    def __init__(self, embed_dim, cond_dim, output_factor=6):
        super().__init__()
        self.output_factor = output_factor
        self.W = nn.Linear(cond_dim, self.output_factor * embed_dim, bias=True)
        # initializing all weights to be 0 is sufficient for initializing the DiT block to be an identity operation (DiT paper claims this could be good empirically)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)
    
    def forward(self, c):
        """
        `c` has shape (batch_size, cond_dim)
        """
        c = F.silu(c)
        c = self.W(c) # (batch_size, output_factor*embed_dim)
        return c.unsqueeze(1).chunk(self.output_factor, dim=-1) # output_factor-tuple where each entry has shape (batch_size, 1, embed_dim)


def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    #cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
    cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    """
    `cos` and `sin` each have shape [1, max_seq_len, 1, head_dim // 2]
    `x` has shape [batch_size, seq_len, num_heads, head_dim]
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2 # head_dim // 2

    # Truncate `cos` and `sin` to the input sequence length
    input_seq_len = x.shape[1]
    cos = cos[:, :input_seq_len, :, :]
    sin = sin[:, :input_seq_len, :, :]

    x1, x2 = x[..., :d], x[..., d:] # split up head dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims (they rotate clockwise, arbitrary choice that I am copying)
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], dim=3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


class BidirectionalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.Wq = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wk = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wv = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.Wo = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.num_heads = config.num_attention_heads
        self.embed_dim = config.embed_dim

    def forward(self, x, cos_sin):
        """
        `x` has shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x) # each shas shape (batch_size, seq_len, embed_dim)

        head_dim = embed_dim // self.num_heads
        assert self.num_heads * head_dim == embed_dim, f"{self.num_heads=}, {head_dim=}, {embed_dim=}"

        q = q.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim) # (batch_size, seq_len, num_heads, head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = rmsnorm(q), rmsnorm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, e.g. (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)

        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1])) # (batch_size, num_heads, seq_len, seq_len)
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.Wo(y)


class DiTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ada_ln_proj = AdaLNProjection(embed_dim=config.embed_dim, cond_dim=config.cond_dim)

        self.attn = BidirectionalSelfAttention(config)
        self.mlp = MLP(input_dim=config.embed_dim, hidden_dim=4*config.embed_dim, output_dim=config.embed_dim)

    def forward(self, x, c, cos_sin):
        """
        `x` has shape (B, L, d) where B is batch size, L is sequence length, and d is embedding dimension
            this is basically a tensor containing sequences of token embeddings 
        `c` has shape (B, d) where B is batch size and d is embedding dimension, this is the time embedding
            which can be viewed as "conditioning" on time (which is why I use the letter "c")
        """
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada_ln_proj(c)

        x = x + gate1 * self.attn( x=F.layer_norm(x, [x.shape[-1]]) * scale1 + shift1, cos_sin=cos_sin ) # TODO maybe do layer norm in fp32?
        x = x + gate2 * self.mlp( F.layer_norm(x, [x.shape[-1]]) * scale2 + shift2 )
        return x


@dataclass
class DiTConfig:
    vocab_size: int = 50304
    embed_dim: int = 768
    cond_dim: int = 768
    sine_base: int = 10000
    num_attention_heads: int = 12
    rotary_base: int = 10000
    num_blocks: int = 12
    max_seq_len: int = 2048

    @classmethod
    def from_dict(cls, d: dict) -> "DiTConfig":
        # treat whole dict as DiTConfig
        return cls(**d)

class DiT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim)
        self.time_emb = TimeEmbedding(input_dim=config.cond_dim, hidden_dim=config.embed_dim, output_dim=config.embed_dim, base=config.sine_base)
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.num_blocks)])
        self.final_ada_ln_proj = AdaLNProjection(embed_dim=config.embed_dim, cond_dim=config.embed_dim, output_factor=2)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        head_dim = config.embed_dim // config.num_attention_heads
        assert config.num_attention_heads * head_dim == config.embed_dim, f"{config.num_attention_heads=}, {head_dim=}, {config.embed_dim=}, {config.embed_dim % config.num_attention_heads=}"

        cos, sin = precompute_rotary_embeddings(seq_len=config.max_seq_len, head_dim=head_dim, base=config.rotary_base)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, token_ids, t):
        """
        `token_ids` has shape (batch_size, seq_len) and is a batch of token id sequences
        `t` has shape (batch_size,) and is a batch of times in [0,1]

        For a given batch index, scores gives a seq_len-by-vocab_size matrix where entry
        (i,j) represents p_t(j)/p_t(token_ids[batch_index][i]) which is only defined if the token
        ids j and token_ids[batch_index][i] are different token ids. 

        Important: We are actually learning the log of the scores here instead of the scores because
        it makes the loss a bit easier to compute. Entries corresponding to the same token id will get
        mapped to 0. This makes the sum excluding the same token in L_DWDSE easier to compute. This
        means that we need to exponentiate the model output when doing inference to obtain the actual
        scores. I took this idea from (for example) here: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/utils.py#L51
        """
        cos_sin = self.cos, self.sin

        c = self.time_emb(t) # (batch_size, embed_dim)
        scale, shift = self.final_ada_ln_proj(c) # 2-tuple of tensors of shape (batch_size, embed_dim) 

        x = self.token_emb(token_ids) # (batch_size, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x=x, c=c, cos_sin=cos_sin)
        x = F.layer_norm(x, [x.shape[-1]]) * scale + shift
        log_scores = self.lm_head(x) # (batch_size, seq_len, vocab_size)

        # Remove scores corresponding to the input token id by setting them to zero
        # torch.scatter() is basically doing a vectorized version of this: 
        # for i in range(batch_size):
        #     for j in range(seq_len):
        #         for k in range(1):  # token_ids[..., None] has shape (batch_size, seq_len, 1)
        #             scores[i][j][token_ids[i][j][k]] = 0
        log_scores = torch.scatter(log_scores, -1, token_ids[..., None], torch.zeros_like(log_scores[..., :1]))

        return log_scores

if __name__ == "__main__":

    config = DiTConfig()
    model = DiT(config)

    batch_size = 2
    token_ids = torch.randint(0, config.vocab_size, (batch_size, 256))
    t = torch.rand((batch_size,))
    print(f"{token_ids.shape=}, {t.shape=}")

    log_scores = model(token_ids=token_ids, t=t)
    print(f"{log_scores.shape=}")

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{params=:,}, {trainable_params=:,}")






