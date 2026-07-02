# Ethon Brain — RunPodサーバレス配信 手順書

HF Space の 1GB 容量制限を回避し、Ethon Brain（自作LLM）を RunPod サーバレスで配信するための
ワーカー一式。**アイドル時は課金ゼロ（scale-to-zero）**、使う時だけ課金。重みはHFデータセット
`latest/` から起動時に自動DL（＝ボリュームが消えても自己回復＝課金切れに強い）。

## 構成
- `handler.py` … RunPodサーバレスのエントリ。起動時に重み自動DL→infer で生成。
- `infer.py` … 純numpy推論（HF Space版と同一）。CPUで動く（GPU不要）。
- `requirements.txt` / `Dockerfile` … ビルド用。
- 重み(weights.npz/vocab.json)は**イメージに焼かず**、起動時に取得。

## デプロイ手順（RunPodコンソール＝要あなたの操作）

RunPodはGitHubからイメージを自動ビルドできるので、ローカルDockerは不要。

1. **このコードをGitHubへ**（staging→本番反映で `mini_llm/runpod_worker/` が上がる）。
2. RunPod → **Serverless → New Endpoint → 「Import from GitHub / Docker」**。
   - GitHubリポジトリを指定し、**Dockerfile のパスを `mini_llm/runpod_worker/Dockerfile`** に。
   - GPUは不要 → **CPUワーカー**（または最小GPU）を選択。
   - Workers: **Min=0 / Max=1**（テスト用途。アイドル課金ゼロ）。
   - Idle timeout: 短め（例5秒）。
3. **Secrets（環境変数）を設定**:
   - `HF_TOKEN` … HFの読み取りトークン（必須。重みDL用）。
   - `HF_DATASET` … `Tgergerbhrc/zibunllm-train`（既定と同じなら省略可）。
4. デプロイ後、**Endpoint ID** をメモ。テストは RunPod の「Requests」タブで
   `{"input":{"question":"変数とは","n":90,"temp":0.6,"mode":"chat"}}` を送信して確認。

## 本体サイト（sensei-gpt.jp）との接続＝要 .env 設定

本体は環境変数が設定された時だけRunPodを呼び、未設定/失敗ならHF Spaceへ自動フォールバック
（＝切替は安全・いつでも戻せる）。本番/テストの `.env` に追記して再起動:

```
MINI_LLM_RUNPOD_ENDPOINT=<RunPodのEndpoint ID>
MINI_LLM_RUNPOD_KEY=<RunPod APIキー>   # 省略時は既存の RUNPOD_API_KEY を流用
```

- 設定 → `/mini_llm` は RunPod 経由（家族限定・公開はしない。既存の家族ウォールのまま）。
- 外すと即HF Spaceに戻る。

## 課金切れ対策（完全自動）
- **本体コピーはHFデータセット `latest/` に常時保管**（無料・RunPod残高0でも消えない）。
- ワーカーは起動時にそこから自動DL → **ボリューム削除されても次回起動で自動復元**。
- 残高監視アラートは `scripts/`（別途）で日次チェック＋RunPodの Auto-Reload 併用推奨。

## 将来のGPU高速化（任意）
現状はCPU推論（335Mで10〜25秒程度）。速くしたい場合は、GPUベースイメージ＋torch移植版の
handlerに差し替える（学習コード `train_gpu.py` のGPTクラスを流用可能）。まずはCPUで容量制限を
解消して配信を確立し、速度は後追いで改善する方針。
