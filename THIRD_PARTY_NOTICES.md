# Third-party notices

## NVIDIA NeMo cache-aware streaming inference

`scripts/run_nemotron.py` contains a streaming inference loop adapted from
NVIDIA NeMo's
[`speech_to_text_cache_aware_streaming_infer.py`](https://github.com/NVIDIA-NeMo/NeMo/blob/e0a284b67db4da54f90a7873dd723aa2106a503e/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py).

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at <https://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## FLEURS-derived evaluation rows

`results/aggregated/worst-cases.csv` and
`results/aggregated/contrast-candidates.csv` contain short reference
transcriptions from the Google FLEURS `ja_jp` test split. FLEURS is licensed
under CC BY 4.0. This benchmark used dataset revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`.

- Source: <https://huggingface.co/datasets/google/fleurs>
- License: <https://creativecommons.org/licenses/by/4.0/>

The audio files themselves are not redistributed by this repository.
