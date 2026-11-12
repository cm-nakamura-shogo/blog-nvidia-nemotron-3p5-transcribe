# Aggregated benchmark results

## Model summary

| Model | Scope | Accuracy runs / BS | Raw CER | Normalized CER | Exact match | Performance runs / BS | Best BS | Audio hours/hour | RTF | Peak VRAM (MiB) | Pareto |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | :---: |
| Whisper large-v3 | main | 1 / 1 | 0.0751 | 0.0449 | 0.3600 | 18 / 1, 4, 8, 16, 32, 64 | 64 | 129.65 | 0.007713 | 20617 | yes |
| Whisper large-v3-turbo | main | 1 / 1 | 0.0815 | 0.0481 | 0.3292 | 18 / 1, 4, 8, 16, 32, 64 | 64 | 210.31 | 0.004755 | 11369 | yes |
| Parakeet TDT 0.6B Japanese | main | 1 / 1 | 0.0941 | 0.0594 | 0.3600 | 18 / 1, 4, 8, 16, 32, 64 | 32 | 2482.36 | 0.000403 | 5629 | yes |
| Kotoba-Whisper v2.2 | main | 1 / 1 | 0.1227 | 0.0692 | 0.2354 | 18 / 1, 4, 8, 16, 32, 64 | 64 | 457.30 | 0.002187 | 6661 | no |
| ReazonSpeech K2 v2 | main | 1 / 1 | 0.1501 | 0.0940 | 0.2400 | 18 / 1, 4, 8, 16, 32, 64 | 32 | 308.84 | 0.003238 | 22661 | no |
| Nemotron offline / Transformers | main | 1 / 1 | 0.1641 | 0.1331 | 0.0600 | 18 / 1, 4, 8, 16, 32, 64 | 64 | 939.18 | 0.001065 | 18467 | no |
| Nemotron streaming / NeMo (ref.) | streaming_reference | 1 / 1 | 0.1642 | 0.1346 | 0.0508 | 18 / 1, 4, 8, 16, 32, 64 | 64 | 573.68 | 0.001743 | 7301 | no |

## Batch scaling

| Model | BS | Runs | Audio hours/hour | RTF | Files/sec | Peak VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| faster-whisper-large-v3 | 1 | 3 | 42.56 | 0.023495 | 3.25 | 4393 |
| faster-whisper-large-v3 | 4 | 3 | 83.54 | 0.011970 | 6.38 | 5193 |
| faster-whisper-large-v3 | 8 | 3 | 104.59 | 0.009561 | 7.99 | 6217 |
| faster-whisper-large-v3 | 16 | 3 | 120.52 | 0.008297 | 9.20 | 8265 |
| faster-whisper-large-v3 | 32 | 3 | 126.51 | 0.007905 | 9.66 | 12425 |
| faster-whisper-large-v3 | 64 | 3 | 129.65 | 0.007713 | 9.90 | 20617 |
| faster-whisper-large-v3-turbo | 1 | 3 | 150.53 | 0.006643 | 11.50 | 2889 |
| faster-whisper-large-v3-turbo | 4 | 3 | 186.25 | 0.005369 | 14.22 | 3273 |
| faster-whisper-large-v3-turbo | 8 | 3 | 197.20 | 0.005071 | 15.06 | 3817 |
| faster-whisper-large-v3-turbo | 16 | 3 | 206.44 | 0.004844 | 15.77 | 4937 |
| faster-whisper-large-v3-turbo | 32 | 3 | 207.62 | 0.004816 | 15.86 | 7049 |
| faster-whisper-large-v3-turbo | 64 | 3 | 210.31 | 0.004755 | 16.06 | 11369 |
| kotoba-whisper-v2.2 | 1 | 3 | 137.45 | 0.007276 | 10.50 | 2825 |
| kotoba-whisper-v2.2 | 4 | 3 | 310.16 | 0.003224 | 23.69 | 3023 |
| kotoba-whisper-v2.2 | 8 | 3 | 387.29 | 0.002582 | 29.58 | 3325 |
| kotoba-whisper-v2.2 | 16 | 3 | 435.44 | 0.002297 | 33.25 | 3683 |
| kotoba-whisper-v2.2 | 32 | 3 | 455.83 | 0.002194 | 34.81 | 4685 |
| kotoba-whisper-v2.2 | 64 | 3 | 457.30 | 0.002187 | 34.92 | 6661 |
| nemotron-3.5-asr-offline-transformers | 1 | 3 | 80.63 | 0.012403 | 6.16 | 3859 |
| nemotron-3.5-asr-offline-transformers | 4 | 3 | 231.91 | 0.004312 | 17.71 | 4671 |
| nemotron-3.5-asr-offline-transformers | 8 | 3 | 389.55 | 0.002567 | 29.75 | 5443 |
| nemotron-3.5-asr-offline-transformers | 16 | 3 | 622.52 | 0.001606 | 47.54 | 7309 |
| nemotron-3.5-asr-offline-transformers | 32 | 3 | 838.75 | 0.001192 | 64.06 | 11025 |
| nemotron-3.5-asr-offline-transformers | 64 | 3 | 939.18 | 0.001065 | 71.73 | 18467 |
| nemotron-3.5-asr-streaming-0.6b | 1 | 3 | 11.93 | 0.083795 | 0.91 | 3739 |
| nemotron-3.5-asr-streaming-0.6b | 4 | 3 | 47.15 | 0.021208 | 3.60 | 3825 |
| nemotron-3.5-asr-streaming-0.6b | 8 | 3 | 92.84 | 0.010772 | 7.09 | 3901 |
| nemotron-3.5-asr-streaming-0.6b | 16 | 3 | 183.08 | 0.005462 | 13.98 | 4145 |
| nemotron-3.5-asr-streaming-0.6b | 32 | 3 | 343.51 | 0.002911 | 26.23 | 4795 |
| nemotron-3.5-asr-streaming-0.6b | 64 | 3 | 573.68 | 0.001743 | 43.81 | 7301 |
| parakeet-tdt-ctc-0.6b-ja | 1 | 3 | 375.51 | 0.002663 | 28.68 | 3743 |
| parakeet-tdt-ctc-0.6b-ja | 4 | 3 | 1211.19 | 0.000826 | 92.50 | 3923 |
| parakeet-tdt-ctc-0.6b-ja | 8 | 3 | 1871.50 | 0.000534 | 142.93 | 4157 |
| parakeet-tdt-ctc-0.6b-ja | 16 | 3 | 2378.96 | 0.000420 | 181.68 | 4649 |
| parakeet-tdt-ctc-0.6b-ja | 32 | 3 | 2482.36 | 0.000403 | 189.58 | 5629 |
| parakeet-tdt-ctc-0.6b-ja | 64 | 3 | 2323.44 | 0.000430 | 177.44 | 7569 |
| reazonspeech-k2-v2 | 1 | 3 | 184.68 | 0.005415 | 14.10 | 2165 |
| reazonspeech-k2-v2 | 4 | 3 | 268.55 | 0.003724 | 20.51 | 5237 |
| reazonspeech-k2-v2 | 8 | 3 | 296.19 | 0.003376 | 22.62 | 6267 |
| reazonspeech-k2-v2 | 16 | 3 | 308.09 | 0.003246 | 23.53 | 10347 |
| reazonspeech-k2-v2 | 32 | 3 | 308.84 | 0.003238 | 23.59 | 22661 |
| reazonspeech-k2-v2 | 64 | 3 | 296.01 | 0.003378 | 22.61 | 41093 |
