# 事前調査ノート

調査日：2026-08-21〜22（Asia/Tokyo）

## 固定した識別子

| 対象 | リポジトリ | revision | パラメータ数 | ライセンス |
| --- | --- | --- | ---: | --- |
| Whisper large-v3（上流） | `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | 1,543,490,560 | Apache-2.0 |
| faster-whisper large-v3変換 | `Systran/faster-whisper-large-v3` | `edaa852ec7e145841d8ffdb056a99866b5f0a478` | 上流と同じ重みのCT2変換 | MIT（変換リポジトリ） |
| Whisper large-v3-turbo（上流） | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 808,878,080 | MIT |
| faster-whisper Turbo変換 | `dropbox-dash/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` | 上流と同じ重みのCT2変換 | MIT |
| Kotoba-Whisper v2.2 | `kotoba-tech/kotoba-whisper-v2.2` | `9d33482a0eb9b57f1ad80708e8ac5538246d8355` | 756,405,760 | Apache-2.0 |
| Parakeet TDT-CTC 0.6B Japanese | `nvidia/parakeet-tdt_ctc-0.6b-ja` | `44edb27eea9317daf89333e75eb830db4b1cc298` | 約600M | CC-BY-4.0 |
| ReazonSpeech K2 v2 | `reazon-research/reazonspeech-k2-v2` | `291488c8151be24d7da4bf7af26e533fad96e407` | 159M | Apache-2.0 |
| Nemotron 3.5 ASR Streaming 0.6B | `nvidia/nemotron-3.5-asr-streaming-0.6b` | `1c8deaecc64b91f034d73e08dd8b64625eb3395d` | 637,997,088（モデルカード表記は600M） | OpenMDW-1.1 |
| FLEURS | `google/fleurs` / `ja_jp` / `test` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | 650件 | CC-BY-4.0 |

パラメータ数はHugging Face Hub APIのsafetensors metadata、revisionはHub APIの`sha`を記録した。記事では丸めた公称値と厳密値を混同しない。

## 比較条件へ影響する確認事項

### Parakeet TDT-CTC 0.6B Japanese

- 日本語専用のHybrid FastConformer TDT-CTCモデルで、約600Mパラメータ。
- ReazonSpeech v2.0の自然な日本語音声35,000時間以上で学習されている。公式モデルカードの学習データ欄にFLEURSは記載されていないが、非包含を証明するものではない。
- TDTはtokenとdurationを同時に予測し、blank計算の多くを飛ばすことで推論高速化を狙う。既定のTDT greedy batch decodeを主結果とする。
- NeMoのHybrid model API上ではTDT経路も`decoder_type=rnnt`として選択する。ロード時ログでRNNT lossが`tdt`、durationが`[0,1,2,3,4]`であることを確認した。
- モデル重みはFP32で保持し、L40S上ではBF16 autocastを使う。外部LM、VAD、タイムスタンプ、外部句読点は無効。
- `transcribe_generator()`をデータセット全体に1回だけ開き、指定batch sizeでyieldされる単位を測定する。各外側batchごとに`transcribe()`を再生成するDataLoaderオーバーヘッドは含めない。

### ReazonSpeech K2 v2

- ReazonSpeech v2.1で公開された日本語専用Zipformer-Transducer。159Mパラメータで、ReazonSpeech v2.0の35,000時間から学習されている。
- 公開重みはFP32/INT8のONNX形式。量子化比較を避けるためFP32のencoder、decoder、joinerを使う。
- 公式パッケージと同じsherpa-onnxのoffline transducer経路を使い、CUDA provider、greedy search、外部LMなしで実行する。
- 公式`transcribe()`は16kHz mono化後の音声の前後へそれぞれ0.9秒の無音を付加する。本実装も同じ無音付加を行い、追加された推論負荷はEnd-to-end時間に含める。スループットの分子は元のFLEURS音声時間とする。
- `OfflineRecognizer.decode_streams()`へ1〜64本を同時に渡してbatch sizeとする。APIは同期的に完了するため、呼び出しの前後をGPU区間の境界とする。
- 公式モデルカードの学習データ欄にFLEURSは記載されていないが、非包含を証明するものではない。

### Nemotron

- 現行revisionはTransformers 5.13以降の`AutoModelForRNNT`によるoffline transcriptionを公式サポートする。録音済みファイルの主比較はこの経路とする。
- NVIDIA Speech NIMのNemotronプロファイルはstreaming onlyであり、その`batch_size=32/64`はoffline APIではなく並列stream処理用である。
- 同一モデルrevisionをTransformers offlineとNeMo cache-aware streamingで利用し、重みの差を入れずに推論経路の差を比較する。
- 日本語`ja-JP`はTranscription-ready tierに含まれる。
- 右コンテキストは`[56, 0]`、`[56, 1]`、`[56, 3]`、`[56, 6]`、`[56, 13]`から選択でき、それぞれ80、160、320、560、1120msに対応する。
- 主結果は精度寄りの`[56, 13]`（1120ms）とする。
- モデルカードはFLEURSを学習データと評価データの両方に明記している。FLEURS上のNemotron精度を未知データへの一般化性能とは表現しない。
- NeMo公式cache-aware inferenceは、モデル全体をBF16へ変換する方式をサポートしていない。モデル重みはFP32で保持し、BF16 autocastを使用する。
- 言語は`target_lang=ja-JP`で固定し、出力の言語タグは除去する。

### Kotoba-Whisper v2.2

- v2.0を基に、話者分離と句読点付与をカスタムpipelineへ統合したリポジトリである。
- ASRコアのv2.0はWhisper large-v3をteacherとする蒸留モデルで、公式カードはReazonSpeech由来の約720万音声クリップを学習データとして説明している。FLEURSは公式カードの学習データ一覧に記載されていない。ただし非包含を証明するものではない。
- 主比較では`trust_remote_code=True`のカスタムpipelineを使わない。
- 同revisionの`WhisperForConditionalGeneration`重みを標準Transformers APIで直接ロードする。
- 話者分離、pyannote、diarizers、punctuatorsを推論経路へ含めない。

### faster-whisper

- `BatchedInferencePipeline.transcribe()`は単一音声を複数チャンクへ分けてバッチ処理する公開APIで、複数の独立ファイルのリストは直接受け取らない。
- 本実装は同pipelineの`forward()`を利用し、各FLEURSファイルから個別に生成した30秒固定長log-Mel特徴量を1つのCTranslate2バッチとして渡す。
- `large-v3-turbo` aliasはfaster-whisperソース上では旧組織名`mobiuslabsgmbh`を指すが、Hub APIは`dropbox-dash/faster-whisper-large-v3-turbo`へリダイレクトする。設定では現在のcanonical IDを固定した。

### FLEURS

- `ja_jp` testは650件。
- FLEURSの`id`列はtest split内で一意ではない。`dataset_index`と`id`を組み合わせたsample IDを作り、650行すべてへ別々のWAVパスを割り当てる。
- Hub上の日本語testデータは約545MB（非圧縮dataset bytesのmetadata値）。固定revisionを取得して16kHz PCM WAVへ変換したmanifestの総音声時間は8,511.12秒（2時間21分51.12秒）。
- Raw CERの正解には`raw_transcription`、Normalized CERには同じ文字列へ共通正規化を適用した値を使う。
- FLEURSが持つ`transcription`列もmanifestへ保存し、評価基準の追跡を可能にする。

### Brev実行環境

- workspace：`cm-nakamura-asr-benchmark`（`l40s-48gb.1x`）
- GPU：NVIDIA L40S、46,068MiB（`nvidia-smi`表示）、driver 565.57.01
- OS：Ubuntu 22.04.5 LTS、Python 3.10.12
- PyTorch：2.11.0+cu129、Transformers 4.57.6、faster-whisper 1.2.1、CTranslate2 4.8.1
- NeMo：`e0a284b67db4da54f90a7873dd723aa2106a503e`（package 3.1.0+e0a284b）
- NemotronのRNNTデコードではフルCUDA Graph構築が`CUDA error: invalid argument`となり、NeMo組み込みのnative PyTorch CUDA Graphへフォールバックした。推論は継続できるが、モデルカードの最適化済み環境と同じ速度とはみなさず記事で明記する。

## 公式出典

- [OpenAI Whisper large-v3 model card](https://huggingface.co/openai/whisper-large-v3)
- [OpenAI Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo)
- [faster-whisper repository](https://github.com/SYSTRAN/faster-whisper)
- [Kotoba-Whisper v2.2 model card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2)
- [NVIDIA Parakeet TDT-CTC 0.6B Japanese model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)
- [ReazonSpeech K2 v2 model repository](https://huggingface.co/reazon-research/reazonspeech-k2-v2)
- [ReazonSpeech v2.1 release note](https://research.reazon.jp/blog/2024-08-01-ReazonSpeech.html)
- [ReazonSpeech official repository](https://github.com/reazon-research/ReazonSpeech)
- [ReazonSpeech K2 official transcribe implementation](https://github.com/reazon-research/ReazonSpeech/blob/master/pkg/k2-asr/src/transcribe.py)
- [ReazonSpeech official audio normalization and padding](https://github.com/reazon-research/ReazonSpeech/blob/master/pkg/k2-asr/src/audio.py)
- [sherpa-onnx Python installation and CUDA provider](https://k2-fsa.github.io/sherpa/onnx/python/install.html)
- [Nemotron 3.5 ASR Streaming 0.6B model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Transformers Nemotron 3.5 ASR documentation](https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/nemotron3_5_asr)
- [NVIDIA Speech NIM Nemotron ASR Streaming documentation](https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/nemotron-asr-streaming.html)
- [NeMo cache-aware streaming inference script](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)
- [FLEURS dataset card](https://huggingface.co/datasets/google/fleurs)
- [FLEURS paper](https://arxiv.org/abs/2205.12446)
- [NVIDIA L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/)
