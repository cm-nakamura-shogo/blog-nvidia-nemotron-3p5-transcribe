# 公式手順の再検証（2026-11-13）

モデルカード（FP8版）の手順をできるだけ変えずに、新しいBrevインスタンスで最初から実行した記録。
失敗した箇所だけを1つずつ最小限に変え、変更点と理由を残す。

## 環境

- Brev workspace: `cm-nakamura-nemotron-transcribe-2`（`l40s-48gb.1x`、新規作成）
- モデル: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8`（実行時点のrevision `2b6209f89bec0da576d69183bfdabb3b6c05ade2`、前回と同じ）
- NeMo Speech: main（実行時点のコミット `079e48770098a4cffcf542a4bc974cafc84502a3`。前回は `e5d79d5`）

## 前提条件

- uvはモデルカードの手順に含まれないため、事前に公式のインストーラで入れる

## 公式のインストール手順（そのまま実行）

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git nemo-speech
cd nemo-speech
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
# PyAV decodes compressed audio formats, such as M4A, without a system FFmpeg.
uv pip install --python .venv/bin/python --no-deps av==19.0.2
# vLLM requires OpenCV, which NeMo's dependency settings exclude.
uv pip install --python .venv/bin/python --no-deps opencv-python-headless==5.0.0.93
source .venv/bin/activate
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 --local-dir ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8
```

モデルカードの注記: FlashInferは初回に量子化Attentionカーネルをコンパイルするため、CUDA 13のnvccが`PATH`に必要（例: `export PATH=/usr/local/cuda/bin:$PATH`）。

## 公式の起動コマンド（FP8版、RTX PRO 6000向け、そのまま実行）

```bash
export VLLM_PLUGINS=nemo_speechlm
# Allow vLLM to decode recordings up to one hour long (the default is 600 seconds).
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
# Word timestamps require the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0
# Keep the optimized CUTLASS FP8 kernels validated on RTX PRO 6000.
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
vllm serve ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.85 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
```

## 前回からわかっている差分

- 前回の起動スクリプトには `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` が抜けていた
- 前回は `--gpu-memory-utilization 0.90`、最大音声長600秒にしていた（必要だったかは未確認）

## 実行記録

（以下、ステップごとに追記）

### ステップ1：公式のインストール手順 → 成功（変更なし）

- uv 0.13.8を公式インストーラで導入（前提条件）
- モデルカードの手順をそのまま実行し、エラーなく完了
- 結果: NeMo Speech `079e487`、Python 3.12.16、vLLM 0.32.0、torch 2.13.0+cu130（CUDA 13.0）、モデル33GB
- Pythonのバージョン: ログ上、`uv venv --python 3.13`で`Using CPython 3.13.18`の環境が作られた後、`uv sync`で`Using CPython 3.12.16` → `Removed virtual environment at: .venv` → `Creating virtual environment at: .venv`と作り直されていた（NeMo Speechの`.python-version`が3.12のため）

### ステップ2：公式の起動コマンド → 失敗（ドライバ）

- `source .venv/bin/activate`、`export PATH=/usr/local/cuda/bin:$PATH`（モデルカードの注記の例）を含めてそのまま実行
- エラー: `RuntimeError: The NVIDIA driver on your system is too old (found version 12070).`（PyTorch 2.13.0+cu130、ドライバ565）

### ステップ3：変更1（cuda-compat-13-0のみ追加） → 起動は成功、最初のリクエストで失敗

- 変更1: `sudo apt-get install -y cuda-compat-13-0`、`export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`
- 起動は成功（`Application startup complete`）。`--gpu-memory-utilization 0.85`、最大音声長1時間のままで、KV cache 422,343トークン
- 前回の問題のうち、次は公式手順どおりなら起きなかった
  - ninjaが見つからない：前回は仮想環境を有効にせず`vllm`を呼んでいた。公式の`source .venv/bin/activate`で解消
  - `curand.h`がない：前回はCUDA 13の最小構成のnvccを使っていた。公式の注記どおり`/usr/local/cuda/bin`（イメージ内のCUDA 12.6のフルツールキット）を使えばヘッダがある
  - gpu-memory-utilization 0.90、最大音声長600秒：不要だった
- サンプル音声（`en-Alice_woman.wav`）で最初のリクエストを送ると、153秒後にHTTP 500
  - `checkpointing_ssu_..._np_9_...`（MTP8）のJITで `ptxas ... error : Instruction 'griddepcontrol' requires .target sm_90 or higher`

### ステップ4a：Mambaをtritonに、MTPは残す → 設定の段階で拒否

- エラー: `Value error, Mamba2 ReplaySSM speculative decoding requires --mamba-backend flashinfer`
- 拒否されているのは「ReplaySSMを使った」投機的デコード。問題のカーネル（`checkpointing_ssu`）もReplaySSM用なので、`--use-replayssm`だけを外す案を次に試す

### ステップ4c：`--use-replayssm`だけを外す（FlashInferとMTPは残す） → 最初のリクエストで失敗

- 起動は成功（KV cache 162,579トークン）
- サンプル音声のリクエストで15秒後にHTTP 500
- エラー: FlashInferの`selective_state_update`（ReplaySSMなしのカーネル）で `tvm.error.InternalError: Error in function 'validate_dtype_consistency'`

### ステップ4b：Mambaをtritonに、MTPを外す → 成功

- 起動成功（FP8 audio encoder 128 Linears、KVキャッシュfp8_e4m3、重み31.08GiB、KV cache 769,185トークン、VRAM 37,005MiB）
- サンプル音声のリクエスト: 1回目3.44秒、2回目0.40秒で正しく書き起こし

## 結論：公式手順からの必要最小限の変更

1. `cuda-compat-13-0`を入れ、`LD_LIBRARY_PATH`に`/usr/local/cuda-13.0/compat`を追加（ドライバ565でPyTorch cu130を動かすため）
2. `--mamba-backend`を`flashinfer`から`triton`に変更（FlashInferのReplaySSMカーネルがsm_90専用命令を含みL40Sでコンパイル不可）
3. `--speculative-config`（MTP）を削除（ReplaySSMの投機的デコードはFlashInfer必須とvLLMが拒否）

上記以外は公式のまま（`source .venv/bin/activate`、`PATH=/usr/local/cuda/bin`、`VLLM_DISABLED_KERNELS`、`--gpu-memory-utilization 0.85`、最大音声長1時間）。

## 計測のやり直し

- 評価用の仮想環境（`setup_brev.sh`）とFLEURSの準備（ja_jp 650件、en_us 647件）は成功
- 精度の計測の最初のリクエストから、すべてHTTP 400（`the JSON object must be str, bytes or bytearray, not tuple`）
  - 直前までサンプル音声のリクエストは200で通っていた。`setup_brev.sh`の実行中（ffmpegなどのapt導入を含む）のどこかでサーバーが壊れた状態になったと思われる
  - `GET /version`をきっかけにする説は、再起動後に再現しなかったため否定
  - サーバーを再起動したら解消（原因は未特定）
- 再起動後、ランナーでFLEURS 10件を確認（CER 1.13%、打ち切り0、空出力0）してから計測を再開

## 実行ごとのばらつき

- 1回目の計測（プローブは同時16件）: terms_verbatim 17/54、terms_context 35/54、terms_verbatim_context 38/54（前回の環境では12、38、38）
- 同じサーバーで同時16件のまま2回追加: 16/36/39、18/36/38。60文中4〜5文の出力が実行ごとに変わる
- 1件ずつ（同時1件）で2回: 2回とも17/36/37で、60文すべて同じ出力
- → 同時に送ると、まとめて処理される組み合わせが変わり、FP8の計算の丸めで出力が揺れると考えられる。プローブの既定の同時実行数を1に変更して、全条件を取り直す
