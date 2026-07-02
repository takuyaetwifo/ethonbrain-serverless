# -*- coding: utf-8 -*-
"""BPE版モデルの純numpy推論(RunPodサーバレス用・HF Space版と同一)。torch非依存。
   weights.npz(重み) と vocab.json(stoi/itos/merges/base_chars + 構成) を同ディレクトリに置く。
   generate / generate_chat / available / info の4関数を提供(HF Space版と同じシグネチャ)。
   ※HF Space の infer.py と同じ内容を保つこと(挙動一致のため)。"""
import numpy as np, json, os, re, threading

_BASE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
_S = {}

# ---- BPEトークナイザ(依存無し) ----
_RE = re.compile(r"[぀-ゟー]+|[゠-・ヽ-ヿ]+|[一-鿿々]+|.", re.DOTALL)
def _pretok(t): return _RE.findall(t)

def available():
    return os.path.exists(os.path.join(_BASE,"weights.npz")) and os.path.exists(os.path.join(_BASE,"vocab.json"))

def _load():
    if _S: return _S
    with _LOCK:
        if _S: return _S
        raw = {k: np.ascontiguousarray(np.asarray(v, dtype=np.float32))
               for k,v in np.load(os.path.join(_BASE,"weights.npz")).items()}
        meta = json.load(open(os.path.join(_BASE,"vocab.json"), encoding="utf-8"))
        stoi = meta["stoi"]; itos = {int(k):v for k,v in meta["itos"].items()}
        rank = {(a,b):i for i,(a,b) in enumerate(meta["merges"])}
        nl, nh, C = meta["n_layer"], meta["n_head"], meta["n_embd"]; hs = C//nh
        layers = []
        for L in range(nl):
            p = f"bk.{L}."
            Wq = np.concatenate([raw[f"{p}sa.h.{h}.q.weight"] for h in range(nh)], axis=0)
            Wk = np.concatenate([raw[f"{p}sa.h.{h}.k.weight"] for h in range(nh)], axis=0)
            Wv = np.concatenate([raw[f"{p}sa.h.{h}.v.weight"] for h in range(nh)], axis=0)
            layers.append(dict(
                l1w=raw[p+"l1.weight"], l1b=raw[p+"l1.bias"],
                l2w=raw[p+"l2.weight"], l2b=raw[p+"l2.bias"],
                WqT=np.ascontiguousarray(Wq.T), WkT=np.ascontiguousarray(Wk.T), WvT=np.ascontiguousarray(Wv.T),
                WpT=np.ascontiguousarray(raw[p+"sa.p.weight"].T), bp=raw[p+"sa.p.bias"],
                ff0T=np.ascontiguousarray(raw[p+"ff.n.0.weight"].T), ff0b=raw[p+"ff.n.0.bias"],
                ff2T=np.ascontiguousarray(raw[p+"ff.n.2.weight"].T), ff2b=raw[p+"ff.n.2.bias"],
            ))
        _S.update(stoi=stoi, itos=itos, rank=rank,
                  block=meta["block"], n_embd=C, n_head=nh, n_layer=nl, hs=hs,
                  scale=1.0/np.sqrt(hs), vocab=meta["vocab"], stop_id=stoi.get("」"),
                  te=raw["te.weight"], pe=raw["pe.weight"],
                  lfw=raw["lf.weight"], lfb=raw["lf.bias"],
                  hdT=np.ascontiguousarray(raw["hd.weight"].T), hdb=raw["hd.bias"],
                  layers=layers)
    return _S

def _merge_word(syms, rank):
    while len(syms) >= 2:
        best, bi = None, None
        for i in range(len(syms)-1):
            r = rank.get((syms[i], syms[i+1]))
            if r is not None and (best is None or r < best): best, bi = r, i
        if bi is None: break
        syms[bi:bi+2] = [syms[bi]+syms[bi+1]]
    return syms

def _encode(S, text):
    stoi, rank = S["stoi"], S["rank"]; ids=[]
    for w in _pretok(text):
        for tok in _merge_word(list(w), rank):
            j = stoi.get(tok)
            if j is not None: ids.append(j)
            else:
                for c in tok:
                    if c in stoi: ids.append(stoi[c])
    return ids

def _ln(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*w+b
def _gelu(x):
    s=np.sign(x); ax=np.abs(x)/np.sqrt(2.0); t=1.0/(1.0+0.3275911*ax)
    erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*np.exp(-ax*ax)
    return 0.5*x*(1+s*erf)
def _softmax(x): x=x-x.max(-1,keepdims=True); e=np.exp(x); return e/e.sum(-1,keepdims=True)

def _step(S, tok, pos, K, V):
    nh, hs, scale = S["n_head"], S["hs"], S["scale"]
    x = S["te"][tok] + S["pe"][pos]
    for i, ly in enumerate(S["layers"]):
        h = _ln(x, ly["l1w"], ly["l1b"])
        q = (h @ ly["WqT"]).reshape(nh, hs)
        k = (h @ ly["WkT"]).reshape(nh, hs)
        v = (h @ ly["WvT"]).reshape(nh, hs)
        K[i].append(k); V[i].append(v)
        Km = np.stack(K[i], axis=1)
        Vm = np.stack(V[i], axis=1)
        scores = np.matmul(Km, q[:, :, None])[:, :, 0] * scale
        att = _softmax(scores)
        ctx = np.matmul(att[:, None, :], Vm)[:, 0, :]
        cat = ctx.reshape(-1)
        x = x + (cat @ ly["WpT"] + ly["bp"])
        h2 = _ln(x, ly["l2w"], ly["l2b"])
        ff = _gelu(h2 @ ly["ff0T"] + ly["ff0b"])
        ff = ff @ ly["ff2T"] + ly["ff2b"]
        x = x + ff
    x = _ln(x, S["lfw"], S["lfb"])
    return x @ S["hdT"] + S["hdb"]

def info():
    S=_load(); return {"vocab":S["vocab"],"block":S["block"],"n_embd":S["n_embd"],"n_layer":S["n_layer"],"tok":"bpe"}

def _sample(logits, temp, rng, recent, rep_pen, top_k, top_p):
    lg=logits.astype(np.float64)/max(temp,1e-6)
    if rep_pen and rep_pen!=1.0 and recent:
        for t in set(recent): lg[t]=lg[t]/rep_pen if lg[t]>0 else lg[t]*rep_pen
    if top_k and 0<top_k<lg.shape[0]:
        kth=np.partition(lg,-top_k)[-top_k]; lg=np.where(lg<kth,-1e30,lg)
    pr=_softmax(lg)
    if top_p and top_p<1.0:
        order=np.argsort(pr)[::-1]; csum=np.cumsum(pr[order]); last=int(np.argmax(csum>=top_p))
        keep=order[:last+1]; mask=np.ones_like(pr,dtype=bool); mask[keep]=False; pr[mask]=0.0
        s=pr.sum(); pr=pr/s if s>0 else _softmax(lg)
    return int(rng.choice(pr.shape[0], p=pr))

def _run(S, ids, n, temp, seed, stop_id=None, top_k=48, top_p=0.92, rep_pen=1.15, rep_window=48):
    block=S["block"]; rng=np.random.default_rng(seed)
    ids=ids[-(block-n-1):] if len(ids)>block-n-1 else ids
    K=[[] for _ in range(S["n_layer"])]
    V=[[] for _ in range(S["n_layer"])]
    logits=None
    for pos,tok in enumerate(ids): logits=_step(S,tok,pos,K,V)
    out=list(ids); g0=len(ids)
    for _ in range(n):
        recent=out[max(g0,len(out)-rep_window):]
        nxt=_sample(logits,temp,rng,recent,rep_pen,top_k,top_p); out.append(nxt)
        if stop_id is not None and nxt==stop_id: break
        logits=_step(S,nxt,len(out)-1,K,V)
    return out

def generate(prompt, n=80, temp=0.8, seed=None):
    S=_load(); itos=S["itos"]; n=max(1,min(int(n),160)); temp=float(min(max(temp,0.2),1.5))
    ids=_encode(S, prompt) or [0]
    out=_run(S, ids, n, temp, seed)
    return "".join(itos.get(i,"") for i in out)

def generate_chat(question, n=90, temp=0.6, seed=None):
    S=_load(); itos=S["itos"]; n=max(1,min(int(n),160)); temp=float(min(max(temp,0.2),1.5))
    q=(question or "").strip()
    ids=_encode(S, f"問「{q}」\n答「")
    if not ids: return ""
    out=_run(S, ids, n, temp, seed, stop_id=S["stop_id"])
    txt="".join(itos.get(i,"") for i in out)
    ans=txt.split("答「",1)[-1]
    if ans.endswith("」"): ans=ans[:-1]
    return ans.strip()
