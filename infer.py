# -*- coding: utf-8 -*-
"""Ethon Brain 推論(torch/GPU版)。CPU純numpy版と同じ関数(available/_load/generate/generate_chat/info)を提供。
   weights.npz(train_gpu.pyのstate_dictをfp16化) と vocab.json を同ディレクトリに置く。
   モデル構造は train_gpu.py と同一(キー名一致=load_state_dictで読める)。GPUがあればcuda、無ければcpu。"""
import os, json, re, threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
_S = {}
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TF32(精度を落とした高速matmul)がAmpere以降のGPUで既定ON。今回のような小型モデルでは
# 累積誤差でsoftmaxの上位トークンが入れ替わり、CPU(numpy/fp32)と全く違う(話題ごとずれた)
# 回答になる不具合が発生した(2026-07-08確認: 同じ重み・同じコードでCPUは正答、GPUは支離滅裂)。
# 常にフルfp32精度で計算させ、CPU版と数学的に一致させる。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_RE = re.compile(r"[぀-ゟー]+|[゠-・ヽ-ヿ]+|[一-鿿々]+|[A-Za-z0-9_]+|.", re.DOTALL)


def available():
    return os.path.exists(os.path.join(_BASE, "weights.npz")) and os.path.exists(os.path.join(_BASE, "vocab.json"))


class _Head(nn.Module):
    def __init__(s, hs, n_embd, block):
        super().__init__(); s.k = nn.Linear(n_embd, hs, bias=False); s.q = nn.Linear(n_embd, hs, bias=False); s.v = nn.Linear(n_embd, hs, bias=False)
        s.register_buffer('t', torch.tril(torch.ones(block, block)))

    def forward(s, x):
        B, T, C = x.shape; k, q, v = s.k(x), s.q(x), s.v(x)
        a = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        a = a.masked_fill(s.t[:T, :T] == 0, float('-inf')); a = F.softmax(a, -1)
        return a @ v


class _MH(nn.Module):
    def __init__(s, nh, hs, n_embd, block):
        super().__init__(); s.h = nn.ModuleList([_Head(hs, n_embd, block) for _ in range(nh)]); s.p = nn.Linear(n_embd, n_embd)

    def forward(s, x):
        return s.p(torch.cat([h(x) for h in s.h], -1))


class _FF(nn.Module):
    def __init__(s, n_embd):
        super().__init__(); s.n = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(s, x):
        return s.n(x)


class _Blk(nn.Module):
    def __init__(s, n_head, n_embd, block):
        super().__init__(); s.sa = _MH(n_head, n_embd // n_head, n_embd, block); s.ff = _FF(n_embd)
        s.l1 = nn.LayerNorm(n_embd); s.l2 = nn.LayerNorm(n_embd)

    def forward(s, x):
        x = x + s.sa(s.l1(x)); x = x + s.ff(s.l2(x)); return x


class _GPT(nn.Module):
    def __init__(s, vocab, n_embd, n_head, n_layer, block):
        super().__init__(); s.te = nn.Embedding(vocab, n_embd); s.pe = nn.Embedding(block, n_embd)
        s.bk = nn.Sequential(*[_Blk(n_head, n_embd, block) for _ in range(n_layer)])
        s.lf = nn.LayerNorm(n_embd); s.hd = nn.Linear(n_embd, vocab)

    def forward(s, idx):
        B, T = idx.shape
        x = s.te(idx) + s.pe(torch.arange(T, device=idx.device))
        x = s.lf(s.bk(x)); return s.hd(x)


def _merge_word(syms, rank):
    while len(syms) >= 2:
        best, bi = None, None
        for i in range(len(syms) - 1):
            r = rank.get((syms[i], syms[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if bi is None:
            break
        syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
    return syms


def _encode(S, text):
    ids = []
    for w in _RE.findall(text):
        for tok in _merge_word(list(w), S["rank"]):
            j = S["stoi"].get(tok)
            if j is not None:
                ids.append(j)
            else:
                for c in tok:
                    if c in S["stoi"]:
                        ids.append(S["stoi"][c])
    return ids


def _load():
    if _S:
        return _S
    with _LOCK:
        if _S:
            return _S
        meta = json.load(open(os.path.join(_BASE, "vocab.json"), encoding="utf-8"))
        stoi = meta["stoi"]; itos = {int(k): v for k, v in meta["itos"].items()}
        rank = {(a, b): i for i, (a, b) in enumerate(meta["merges"])}
        z = np.load(os.path.join(_BASE, "weights.npz"))
        vocab = int(z["te.weight"].shape[0]); n_embd = int(z["te.weight"].shape[1])
        block = int(z["pe.weight"].shape[0])
        n_layer = 1 + max(int(m.group(1)) for k in z.files for m in [re.match(r"bk\.(\d+)\.", k)] if m)
        n_head = 1 + max(int(m.group(1)) for k in z.files for m in [re.match(r"bk\.0\.sa\.h\.(\d+)\.", k)] if m)
        model = _GPT(vocab, n_embd, n_head, n_layer, block)
        sd = {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files}
        model.load_state_dict(sd, strict=False)
        model.eval().to(_DEVICE)
        _S.update(model=model, stoi=stoi, itos=itos, rank=rank, block=block, vocab=vocab,
                  n_embd=n_embd, n_head=n_head, n_layer=n_layer, stop_id=stoi.get("」"))
    return _S


def info():
    S = _load()
    return {"vocab": S["vocab"], "block": S["block"], "n_embd": S["n_embd"], "n_layer": S["n_layer"], "tok": "bpe", "device": _DEVICE}


@torch.no_grad()
def _run(S, ids, n, temp, seed, stop_id=None, top_k=48, top_p=0.92, rep_pen=1.15, rep_window=48):
    block = S["block"]; model = S["model"]
    if seed is not None:
        torch.manual_seed(int(seed))
    ids = ids[-(block - n - 1):] if len(ids) > block - n - 1 else ids
    out = list(ids); g0 = len(out)
    idx = torch.tensor([out], dtype=torch.long, device=_DEVICE)
    for _ in range(n):
        logits = model(idx[:, -block:])[:, -1, :].float().squeeze(0) / max(temp, 1e-6)
        if rep_pen and rep_pen != 1.0:
            recent = out[max(g0, len(out) - rep_window):]
            for t in set(recent):
                logits[t] = logits[t] / rep_pen if logits[t] > 0 else logits[t] * rep_pen
        if top_k and 0 < top_k < logits.shape[0]:
            kth = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, -1e30), logits)
        probs = F.softmax(logits, -1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, 0)
            cut = int(torch.searchsorted(csum, torch.tensor(float(top_p), device=_DEVICE)).item()) + 1
            keep = si[:cut]
            mask = torch.ones_like(probs, dtype=torch.bool); mask[keep] = False
            probs = probs.masked_fill(mask, 0.0)
            s = probs.sum()
            probs = probs / s if s > 0 else F.softmax(logits, -1)
        nxt = int(torch.multinomial(probs, 1).item())
        out.append(nxt)
        if stop_id is not None and nxt == stop_id:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long, device=_DEVICE)], 1)
    return out


def generate(prompt, n=80, temp=0.8, seed=None):
    S = _load(); itos = S["itos"]; n = max(1, min(int(n), 160)); temp = float(min(max(temp, 0.2), 1.5))
    ids = _encode(S, prompt) or [0]
    out = _run(S, ids, n, temp, seed)
    return "".join(itos.get(i, "") for i in out)


def generate_chat(question, n=90, temp=0.6, seed=None):
    S = _load(); itos = S["itos"]; n = max(1, min(int(n), 160)); temp = float(min(max(temp, 0.2), 1.5))
    q = (question or "").strip()
    ids = _encode(S, f"問「{q}」\n答「")
    if not ids:
        return ""
    out = _run(S, ids, n, temp, seed, stop_id=S["stop_id"])
    txt = "".join(itos.get(i, "") for i in out)
    ans = txt.split("答「", 1)[-1]
    if ans.endswith("」"):
        ans = ans[:-1]
    return ans.strip()
