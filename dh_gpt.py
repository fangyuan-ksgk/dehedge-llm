"""GPT and DeHedgeGPT — a compact decoder-only LM and its de-hedged variant.

  GPT          : standard pre-LN transformer, tied unembedding (the next-token-prediction baseline).
  DeHedgeGPT   : GPT + a marginal BIAS CHANNEL (additive log-unigram output bias). Combine with the ISOTROPY
                 regularizer (`isotropy.iso_loss` on the returned final hidden state) at train time = "de-hedging".

`forward(idx, return_rep=True)` also returns the final hidden state so the trainer can apply `iso_loss(rep)`.
State-dict layout matches the original tinystories_gpt.GPT, so existing checkpoints load unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Pre-LN transformer block (multi-head causal attention + GELU MLP)."""
    def __init__(self, ne, nh):
        super().__init__()
        self.ln1 = nn.LayerNorm(ne)
        self.ln2 = nn.LayerNorm(ne)
        self.nh = nh
        self.qkv = nn.Linear(ne, 3 * ne)
        self.proj = nn.Linear(ne, ne)
        self.up = nn.Linear(ne, 4 * ne)
        self.act = nn.GELU()
        self.down = nn.Linear(4 * ne, ne)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(a)
        x = x + self.down(self.act(self.up(self.ln2(x))))
        return x


class GPT(nn.Module):
    """Decoder-only LM. Set bias_channel=True (or use DeHedgeGPT) to add the marginal bias channel."""
    def __init__(self, vocab, ne=384, nl=6, nh=6, block=256, bias_channel=False, ubias_init=None):
        super().__init__()
        self.block = block
        self.wte = nn.Embedding(vocab, ne)
        self.wpe = nn.Embedding(block, ne)
        self.h = nn.ModuleList([Block(ne, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(ne)
        self.lm_head = nn.Linear(ne, vocab, bias=False)
        self.wte.weight = self.lm_head.weight  # tie
        if bias_channel:
            init = ubias_init if ubias_init is not None else torch.zeros(vocab)
            self.ubias = nn.Parameter(init.clone())
        else:
            self.ubias = None

    def forward(self, idx, targets=None, return_rep=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.h:
            x = b(x)
        rep = self.lnf(x)                      # final hidden state (what the readout sees)
        logits = self.lm_head(rep)
        if self.ubias is not None:
            logits = logits + self.ubias       # marginal carried OUTSIDE the representation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_rep:
            return logits, loss, rep
        return logits, loss


def DeHedgeGPT(vocab, ubias_init=None, **kw):
    """GPT with the marginal bias channel enabled. Train with isotropy.iso_loss on the returned rep for full de-hedging."""
    return GPT(vocab, bias_channel=True, ubias_init=ubias_init, **kw)
