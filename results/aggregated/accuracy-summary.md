# Accuracy results

| Model | Scope | Runs | BS | Raw CER mean | Normalized CER mean | Normalized CER range | Std | Exact match | Empty output |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper large-v3 | main | 1 | 1 | 0.0751 | 0.0449 | 0.0449–0.0449 | 0.0000 | 0.3600 | 0.0000 |
| Whisper large-v3-turbo | main | 1 | 1 | 0.0815 | 0.0481 | 0.0481–0.0481 | 0.0000 | 0.3292 | 0.0000 |
| Parakeet TDT 0.6B Japanese | main | 1 | 1 | 0.0941 | 0.0594 | 0.0594–0.0594 | 0.0000 | 0.3600 | 0.0000 |
| Kotoba-Whisper v2.2 | main | 1 | 1 | 0.1227 | 0.0692 | 0.0692–0.0692 | 0.0000 | 0.2354 | 0.0000 |
| ReazonSpeech K2 v2 | main | 1 | 1 | 0.1501 | 0.0940 | 0.0940–0.0940 | 0.0000 | 0.2400 | 0.0000 |
| Nemotron offline / Transformers | main | 1 | 1 | 0.1641 | 0.1331 | 0.1331–0.1331 | 0.0000 | 0.0600 | 0.0000 |
| Nemotron streaming / NeMo (ref.) | streaming_reference | 1 | 1 | 0.1642 | 0.1346 | 0.1346–0.1346 | 0.0000 | 0.0508 | 0.0000 |
