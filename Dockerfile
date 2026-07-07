# Ethon Brain RunPodサーバレス・ワーカー(torch/GPU推論)。
# GPUがあればcuda、無ければCPUで動く(torchが自動判定)。重みは起動時にHF datasetのlatest/から取得。
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py handler.py ./

CMD ["python", "-u", "handler.py"]
