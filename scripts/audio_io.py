"""Audio loading helpers shared by inference runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


def load_audio_mono(path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        waveform = resample_poly(
            waveform,
            sample_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)

