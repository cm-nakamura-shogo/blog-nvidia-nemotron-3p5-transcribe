# Nemotron 3.5 Transcribe 日本語検証

[NVIDIA Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)（2026-10-28公開）を、
NVIDIA L40S 1基で日本語の音声に対して検証したコードと集計結果です。
DevelopersIOの記事「Nemotron 3.5 Transcribeを日本語で試したら、専門用語の認識精度がプロンプトだけで22%から70%に上がった」で使用しました。

以前の[日本語ASR比較ベンチマーク](#日本語asr比較ベンチマーク)のハーネスに、vLLMのOpenAI互換サーバーへリクエストを送るランナーを追加しています。

| 観点 | 内容 | 主なファイル |
| --- | --- | --- |
| 専門用語リスト | Nemotron 3.5 ASRのファインチューニング記事の台本60文（専門用語を含む54文と対照の6文）で、`Context terms:`の効果を測定 | `configs/prompt-probes.yaml`, `data/discovery/` |
| 表記の指示 | FLEURS `ja_jp` testで、数字の表記と句読点の指示を比較 | `configs/prompt-probes.yaml`, `scripts/analyze_format_probes.py` |
| 音声からの直接翻訳 | FLEURS testの日→英、英→日。参照訳は同じ`fleurs_id`の他言語の書き起こし | `configs/prompt-probes.yaml`, `scripts/score_translation.py` |
| 認識精度 | FLEURS `ja_jp` test 650件の正規化CER | `configs/nemotron-3.5-transcribe-fp8-vllm.yaml`, `scripts/run_nemotron_transcribe_vllm.py` |
| 処理時間（参考） | 1件ずつ処理したときのリクエスト単位の処理時間 | 同上 |

## 比較条件の注意

- モデルカードはFLEURSを学習データに明記しているため、FLEURSの精度は汎化性能として解釈しません。
- モデルカードの数値はBF16版です。FP8版は文字起こしの誤り率が相対で平均1.0%高いと記載されています。
- L40SではMTP（投機的デコード）を使えないため、処理速度はほかのモデルと比較していません。記事では、1件ずつ処理したときの処理時間だけを参考として載せています。
- 同時に複数のリクエストを送ると、temperature 0でも出力が実行ごとに数件変わりました（60文中4〜5文）。1件ずつ送ると毎回同じ出力になったため、すべての計測を1件ずつで行っています（`results/brev/probes-concurrent16/`、`probes-rep2/`、`probes-rep3/`、`probes-c1a/`、`probes-c1b/`はその確認の結果です）。
- 専門用語リストは、ファインチューニング記事のPhrase Boostingと同じ25語です（台本の対象語のうちCTCとDGXは含みません）。
- 台本（`data/discovery/`）はファインチューニング記事のものです。録音（`work/recordings/discovery/`）はリポジトリに含めません。

## L40S向けの変更

インストールと起動は、モデルカード（FP8版）の手順をできるだけそのまま使い、L40Sで動かすために必要な3か所だけを変えています。
どの変更が必要かは、新しいBrevインスタンスで公式手順をそのまま実行し、失敗した箇所を1つずつ直して確かめました（記録は`verification/official-rerun.md`、各段階の起動スクリプトは`results/brev/official-serve/`）。

| 変更 | 理由 |
| --- | --- |
| `cuda-compat-13-0`を入れ、`LD_LIBRARY_PATH`に`/usr/local/cuda-13.0/compat`を追加 | Brevの`l40s-48gb.1x`はドライバ565（CUDA 12.7）で、vLLM 0.32.0が入れるtorch 2.13.0+cu130が動かないため。forward compatibilityパッケージでCUDA 13のランタイムを使う |
| `--mamba-backend`を`flashinfer`から`triton`に変更 | FlashInfer 0.7.0.post1のReplaySSMカーネル（`checkpointing_ssu`）がsm_90専用命令（`griddepcontrol`）をGPUの世代による分岐なしで含み、L40S（sm_89）では最初のリクエストでJITコンパイルに失敗するため |
| MTPの投機的デコード（`--speculative-config`）を削除 | vLLMが`Mamba2 ReplaySSM speculative decoding requires --mamba-backend flashinfer`として拒否するため |

`--use-replayssm`だけを外してFlashInferとMTPを残す構成も試しましたが、FlashInferの別のカーネル（`selective_state_update`）でデータ型のエラーになりました。

それ以外（`source .venv/bin/activate`、`PATH=/usr/local/cuda/bin`、`VLLM_DISABLED_KERNELS`、`--gpu-memory-utilization 0.85`、最大音声長1時間）は、モデルカードのままです。
インストールは`scripts/setup_nemotron_transcribe.sh`、起動は`scripts/serve_nemotron_transcribe.sh`です。

## 実行（Brev L40S）

```bash
export ASR_BENCHMARK_BREV_WORKSPACE=cm-nakamura-nemotron-transcribe-2
bash scripts/host/create_workspace.sh
bash scripts/host/sync_to_brev.sh
# Brev内
bash scripts/setup_brev.sh
bash scripts/setup_nemotron_transcribe.sh
bash scripts/brev/run_transcribe_all.sh
```

`run_transcribe_all.sh`は、FLEURSの準備、vLLMサーバーの起動、認識精度の測定、プロンプトを変えた実行と採点を順に実行します。

## 集計結果

2026-11-13に測定した結果を`results/brev/`に置いています。

| パス | 内容 |
| --- | --- |
| `results/brev/accuracy/` | 認識精度の測定のメタデータ（`*.meta.json`） |
| `results/brev/probes/` | プロンプトの条件ごとのメタデータ、専門用語の評価（`*-term-summary.json`、`*-term-details.csv`）、表記の集計（`format-summary.json`）、翻訳のスコア（`*-score.json`） |
| `results/brev/environment/` | 実行環境のパッケージ一覧（`pip-freeze-official-nemo-speech.txt`がvLLMの環境） |
| `results/brev/official-serve/` | 公式手順を段階的に試したときのインストールと起動のスクリプト |

発話ごとの認識結果（JSONL）とログは、容量と再配布範囲を考慮してリポジトリに含めていません。

---

# 日本語ASR比較ベンチマーク

NVIDIA L40S 1基で、FLEURS `ja_jp` testを用いて次の「モデル＋ランタイム」構成を比較するための再現コードです。

- Whisper large-v3 + faster-whisper/CTranslate2
- Whisper large-v3-turbo + faster-whisper/CTranslate2
- Kotoba-Whisper v2.2 + Transformers（ASR重みのみ）
- NVIDIA Parakeet TDT-CTC 0.6B Japanese + NVIDIA NeMo
- ReazonSpeech K2 v2 + sherpa-onnx CUDA
- Nemotron 3.5 ASR Streaming 0.6B + Transformers offline RNNT
- Nemotron 3.5 ASR Streaming 0.6B + NVIDIA NeMo cache-aware streaming（参考）

記事本文は[`article/benchmark-notes.md`](article/benchmark-notes.md)、事前調査の記録は[`article/research-notes.md`](article/research-notes.md)です。

## 重要な比較条件

- FLEURSの音声ファイルと入力順は全構成で共通です。
- FLEURSの`id`列はsplit内で一意ではないため、`dataset_index`と`id`を組み合わせたsample IDとWAV名を使います。準備処理はIDと音声パスの一意性を検査します。
- 準備後は`validate_manifest.py`で650 WAVの存在、16kHz mono、manifest記載時間、SHA-256の一意性を検査し、結果を`manifests/fleurs-ja_jp-test.validation.json`へ保存します。
- 主比較はgreedy decode、言語固定、VADなし、タイムスタンプなし、外部後処理なしです。
- Warm計測はモデルロードとwarm-upを除き、音声読み込みから文字列取得までをEnd-to-end時間とします。
- `bucket`順では音声時間の降順、同じ長さなら`sample_id`順で固定します。
- NemotronのモデルカードはFLEURSを学習データに明記しています。NemotronのFLEURS精度は未知データへの汎化性能として解釈しません。
- Nemotronは同一revisionの重みをTransformersのoffline RNNTとNeMoのcache-aware streamingで実行し、モデル差ではなく推論経路の差を分離します。
- ParakeetはNeMoのTDT greedy batch decodeを使い、FP32重みをL40S上のBF16 autocastで推論します。
- ReazonSpeechは公式実装と同じFP32 ONNX重みとCUDA providerを使い、入力の前後にそれぞれ0.9秒の無音を付加します。
- Kotoba v2.2のカスタムpipelineは話者分離と句読点後処理を統合します。主比較では標準の`AutoModelForSpeechSeq2Seq`で同じリポジトリのASR重みだけをロードします。
- faster-whisperの公開`BatchedInferencePipeline.transcribe()`は1音声内の複数チャンクを対象にします。このコードは同pipelineの`forward()`を用い、独立したFLEURS音声から作ったlog-Mel特徴量を1つのCTranslate2バッチへまとめます。

## ディレクトリ

```text
benchmark/
├── configs/
├── environment/
├── manifests/
├── results/
│   ├── raw/
│   ├── aggregated/
│   └── figures/
├── scripts/
├── tests/
└── article/
```

データセット本体とモデル重みはリポジトリへ保存しません。

リポジトリ全体のライセンスはMITです。NeMo公式コードを参照した部分の著作権とApache-2.0通知は[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)に記載しています。

集計後は入力に応じて、Accuracyの`accuracy-summary.csv/json/md`と`accuracy-runs.csv`、Performanceの`performance-summary.csv/json`と`performance-runs.csv`、`by-batch.csv`を生成します。両方を入力した場合は、`model-summary.csv/json`、`summary.md`、Pareto図も生成します。Accuracy入力時は`worst-cases.csv`と、7構成がそろった場合の`contrast-candidates.csv`も作成します。生の音声別JSONLは容量と再配布範囲を考慮してGitHubには含めません。

## L40S実測結果

2026-08-21〜22にFLEURS `ja_jp` test 650件を使って測定した結果です。精度はBS1、速度は各構成の最高スループットです。

| 構成 | Normalized CER | 最高Audio hours/hour | BS | 最大VRAM |
| --- | ---: | ---: | ---: | ---: |
| Whisper large-v3 | 4.49% | 129.6 | 64 | 20.13GiB |
| Whisper large-v3-turbo | 4.81% | 210.3 | 64 | 11.10GiB |
| Parakeet TDT 0.6B Japanese | 5.94% | **2,482.4** | 32 | **5.50GiB** |
| Kotoba-Whisper v2.2 | 6.92% | 457.3 | 64 | 6.50GiB |
| ReazonSpeech K2 v2 | 9.40% | 308.8 | 32 | 22.13GiB |
| Nemotron offline / Transformers | 13.31% | 939.2 | 64 | 18.03GiB |
| Nemotron streaming / NeMo（参考） | 13.46% | 573.7 | 64 | 7.13GiB |

7構成×6バッチサイズ×3回の126 runはすべて完了し、81,900件の予測をmanifestと照合済みです。OOMと空出力はありませんでした。主比較のPareto frontierにはWhisper large-v3、large-v3-turbo、Parakeetが残りました。詳細は[`results/aggregated/summary.md`](results/aggregated/summary.md)と[`article/benchmark-notes.md`](article/benchmark-notes.md)を参照してください。

## Brevでの実行

専用のBrev workspace `cm-nakamura-asr-benchmark`（`l40s-48gb.1x`）を利用します。

```bash
cd benchmark
bash scripts/host/create_workspace.sh
bash scripts/host/sync_to_brev.sh
bash scripts/host/run_on_brev.sh
```

Brev内では次の順で実行します。

```bash
cd /home/ubuntu/workspace/asr-benchmark
bash scripts/setup_brev.sh
bash scripts/setup_nemotron_transformers.sh
bash scripts/setup_reazonspeech.sh
source work/.venv/bin/activate

python scripts/prepare_fleurs.py \
  --audio-dir work/fleurs-ja_jp-test/audio \
  --manifest manifests/fleurs-ja_jp-test.jsonl \
  --metadata manifests/fleurs-ja_jp-test.metadata.json
python scripts/validate_manifest.py \
  --manifest manifests/fleurs-ja_jp-test.jsonl \
  --output manifests/fleurs-ja_jp-test.validation.json \
  --expected-files 650 \
  --expected-sample-rate 16000
bash scripts/brev/03_smoke_test.sh
bash scripts/brev/04_run_accuracy.sh
bash scripts/brev/05_run_performance.sh
bash scripts/brev/06_aggregate.sh \
  --accuracy-dir results/accuracy \
  --accuracy-status-dir results/status-accuracy \
  --performance-dir results/raw \
  --performance-status-dir results/status
bash scripts/brev/07_collect_environment.sh
```

`06_aggregate.sh`はAccuracyとPerformanceの入力ディレクトリを対称に受け取り、少なくともどちらか一方を指定します。集計対象は、指定したフォルダに実在する正常なrunだけです。

- AccuracyはBS1を必須とせず、モデルごとに実在runの認識指標を平均します。CERの最小、最大、標準偏差も出力します。
- Performanceは実在するモデル×BSごとに、実在する反復runの中央値を出力します。全6BS・3回がそろっている必要はありません。Performance側でCERは再計算しません。
- 各runのモデルrevision、650件のsample ID集合と一意性を検証し、`accuracy-verification.json`と`performance-verification.json`へ保存します。statusディレクトリは任意で、指定時は`completed`も照合します。

精度結果だけを集計する場合はPerformanceの2引数を省略し、出力先を分けます。

```bash
bash scripts/brev/06_aggregate.sh \
  --accuracy-dir results/accuracy \
  --accuracy-status-dir results/status-accuracy \
  --output-dir results/aggregated-accuracy \
  --figure-dir results/figures-accuracy
```

性能結果だけを集計する場合も同様です。例えば指定フォルダにBS 8と16だけがある場合、その2条件だけを集計します。

```bash
bash scripts/brev/06_aggregate.sh \
  --performance-dir results/raw-partial \
  --performance-status-dir results/status-partial \
  --output-dir results/aggregated-performance \
  --figure-dir results/figures-performance
```

`--output-dir`と`--figure-dir`のデフォルトは、それぞれ`results/aggregated`と`results/figures`です。部分集計では、既存出力との混在を避けるため別の出力先を推奨します。予定したmatrixの完了自体を監査するときは、`verify_results.py`を`--available-only`なしで別途実行します。

Pythonの集計処理を直接実行する場合も、AccuracyとPerformanceを別引数で指定します。

```bash
python scripts/aggregate_results.py \
  --accuracy-input-dir results/accuracy \
  --performance-input-dir results/raw \
  --output-dir results/aggregated \
  --figure-dir results/figures
```

長時間の性能測定はSSH切断の影響を受けないよう、次のランチャーでBrev内のバックグラウンドジョブとして実行できます。既存の完了条件は`--resume`で再利用されます。

```bash
bash scripts/brev/start_performance_background.sh
tail -f results/full-benchmark.log
```

NemotronのTransformers offline構成だけを分離して実行する場合は次を使います。

```bash
bash scripts/brev/start_nemotron_offline_background.sh
tail -f results/nemotron-offline-benchmark.log
```

ParakeetとReazonSpeechの追試だけを再開可能なバックグラウンドジョブで実行する場合は次を使います。

```bash
bash scripts/brev/start_additional_models_background.sh
tail -f results/additional-models-benchmark.log
```

実行終了後は結果をローカルへ戻し、workspaceを停止します。

```bash
bash scripts/host/fetch_results.sh
brev stop cm-nakamura-asr-benchmark
```

## ローカル単体テスト

GPUやモデルを使わない正規化・CER・集計処理はローカルで確認できます。

```bash
python3 -m unittest discover -s tests -v
```
