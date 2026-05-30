"""Description-init embeddings.

Build a [vocab, n_embd] embedding-init matrix from a token->vector file produced by
`scripts/build_emb_init.py` (a frozen text encoder over each step's name/description).
Tokens that aren't in the file (the specials, or a truly-unseen token) fall back to
random init. The whole matrix is scaled to the usual init std so training dynamics
are unchanged — this is a *warm start*, not a frozen embedding.

Why it matters: when a family is held out (OOD), its distinctive step tokens otherwise
keep random embeddings (no gradient). Seeding them from their *text* places them near
semantically-related trained steps, so the model can rank them and their presence in a
prefix no longer corrupts the context. Pure numpy — unit-testable without torch.
"""
from __future__ import annotations

import numpy as np


def build_embedding_init(vocab, npz_path, n_embd, seed=0, scale=0.02):
    """Return (init[vocab, n_embd] float32, n_semantic_tokens).

    - L2-normalize each encoder vector, project enc_dim -> n_embd with a fixed seeded
      Gaussian map (Johnson-Lindenstrauss: approximately preserves angles/cosine), then
      rescale the whole matrix to std == `scale`.
    - Missing tokens (specials / unseen) get random N(0, scale) rows.
    """
    data = np.load(npz_path, allow_pickle=True)
    toks = [str(t) for t in data["tokens"]]
    vecs = np.asarray(data["vectors"], dtype=np.float64)          # [n, enc_dim]
    enc_dim = vecs.shape[1]
    tok2vec = {t: vecs[i] for i, t in enumerate(toks)}

    rng = np.random.default_rng(seed)
    R = rng.standard_normal((enc_dim, n_embd)) / np.sqrt(enc_dim)  # fixed projection

    out = np.empty((len(vocab), n_embd), dtype=np.float64)
    n_sem = 0
    for i, t in enumerate(vocab.itos):
        v = tok2vec.get(t)
        if v is None:
            out[i] = rng.standard_normal(n_embd)                  # specials / unseen -> random
        else:
            out[i] = (v / (np.linalg.norm(v) + 1e-8)) @ R         # normalized + projected
            n_sem += 1

    out = out / (out.std() + 1e-8) * scale                        # match default init scale
    return out.astype(np.float32), n_sem
