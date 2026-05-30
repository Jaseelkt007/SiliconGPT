"""Decoder-only transformer for process-step sequences.

Modern components: RMSNorm, rotary positional embeddings (RoPE), SwiGLU MLP,
weight-tied head, PyTorch scaled_dot_product_attention (Flash on GPU).

Padding note: we use right-padding + causal attention, so a real token never
attends to a (future) pad token, and pad positions carry label=-100 and are
ignored in the loss. Hence no explicit key-padding mask is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 202
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.1
    mlp_ratio: float = 8.0 / 3.0
    rope_base: float = 10000.0
    tie_weights: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).type_as(self.weight)


def build_rope_cache(head_dim, max_seq, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)     # [T, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, T, head_dim]; cos/sin: [T, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:T].to(q.dtype), sin[:T].to(q.dtype))
        k = apply_rope(k, cos[:T].to(k.dtype), sin[:T].to(k.dtype))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        hidden = (hidden + 7) // 8 * 8
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd, hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class ProcessLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        cos, sin = build_rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        assert T <= self.cfg.block_size, f"seq len {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok(input_ids))
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
