# Ethon Brain RunPodサーバレス・ワーカー(CPU推論・純numpy)。
# GPUは不要(モデルはnumpy)。将来GPU高速化する場合はGPUベースイメージ＋torch移植に差し替える。
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py handler.py ./

# 重み(weights.npz/vocab.json)はイメージに焼かず、起動時にHFデータセットから取得する
# (self-heal・イメージを軽く保つ)。HF_TOKEN は RunPod の Secrets で渡す。
CMD ["python", "-u", "handler.py"]
