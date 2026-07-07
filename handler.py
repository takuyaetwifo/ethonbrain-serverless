# -*- coding: utf-8 -*-
"""Ethon Brain の RunPodサーバレス・ハンドラ。
   ・起動時に HFデータセット(latest/weights.npz + latest/vocab.json)を infer.py の隣へ自動DL
     (＝RunPodのボリュームが消えても、次回起動で自動復元＝自己回復。課金切れに強い)。
   ・リクエスト {input:{question|prompt, n, temp, mode}} を受け、infer で生成して返す。
   秘密は env(RunPodのSecretsで設定): HF_TOKEN(必須) / HF_DATASET(既定 Tgergerbhrc/zibunllm-train)。
"""
import os, shutil, traceback
import runpod

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DATASET = os.environ.get("HF_DATASET", "Tgergerbhrc/zibunllm-train")


def _ensure_weights():
    """weights.npz / vocab.json が無ければ HFデータセットの latest/ から取得(自己回復)。"""
    import infer
    if infer.available():
        return
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN") or None
    for remote, local in (("latest/weights.npz", "weights.npz"),
                          ("latest/vocab.json", "vocab.json")):
        path = hf_hub_download(HF_DATASET, remote, repo_type="dataset", token=token)
        dst = os.path.join(HERE, local)
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy(path, dst)


# コールドスタート時に一度だけ実行(重みの準備＋モデルのメモリ常駐)。
try:
    _ensure_weights()
    import infer
    infer._load()   # 起動時にロードしておき、初回応答を速く
    _READY = True
    _INIT_ERR = ""
except Exception as e:
    _READY = False
    _INIT_ERR = f"{e}\n{traceback.format_exc()}"


def handler(event):
    if not _READY:
        return {"error": "model not ready", "detail": _INIT_ERR[:500]}
    inp = (event or {}).get("input", {}) or {}
    q = (inp.get("question") or inp.get("prompt") or "").strip()
    if not q:
        return {"error": "empty question/prompt"}
    try:
        n = int(inp.get("n", 90))
    except (TypeError, ValueError):
        n = 90
    try:
        temp = float(inp.get("temp", 0.6))
    except (TypeError, ValueError):
        temp = 0.6
    mode = (inp.get("mode") or "chat").lower()
    if mode == "echo":
        # 診断用(2026-07-08): 実際にRunPod経由で届く質問文字列がどう見えるか確認する。
        import infer as _dbg
        _dbg._load()
        ids = _dbg._encode(_dbg._load(), f"問「{q}」\n答「")
        return {"received_question": q, "len": len(q), "codepoints": [hex(ord(c)) for c in q],
                "encoded_ids": ids, "decoded_back": "".join(_dbg._load()["itos"].get(i, "?") for i in ids)}
    import infer
    try:
        if mode in ("continue", "generate"):
            return {"text": infer.generate(q, n=n, temp=temp), "engine": "runpod"}
        return {"answer": infer.generate_chat(q, n=n, temp=temp), "engine": "runpod"}
    except Exception as e:
        return {"error": "inference failed", "detail": str(e)[:300]}


runpod.serverless.start({"handler": handler})
