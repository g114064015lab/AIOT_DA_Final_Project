"""
Generate sample audio clips for quick testing of the Streamlit demo.
Produces:
- samples/demo_gunshot_glass.wav
- samples/demo_noise.wav
"""

from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf


def generate_demo_audio(duration: float = 8.0, sample_rate: int = 16000):
    rng = np.random.default_rng(0xC0DE)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise = rng.normal(0, 0.02, size=t.shape)

    gunshot_center = int(0.35 * len(t))
    gunshot = np.zeros_like(t)
    gunshot[gunshot_center : gunshot_center + 300] = 1.0
    gunshot = np.convolve(gunshot, np.hanning(120), mode="same")

    glass_center = int(0.72 * len(t))
    glass = np.zeros_like(t)
    glass[glass_center : glass_center + 600] = np.sin(2 * np.pi * 5500 * t[:600]) * np.hanning(600)

    y = noise + 0.8 * gunshot + 0.4 * glass
    y = y / np.abs(y).max()
    return y.astype(np.float32), sample_rate


def generate_noise(duration: float = 10.0, sample_rate: int = 16000):
    rng = np.random.default_rng(1234)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise = rng.normal(0, 0.05, size=t.shape)
    hum = 0.1 * np.sin(2 * np.pi * 60 * t)
    y = noise + hum
    y = y / np.abs(y).max()
    return y.astype(np.float32), sample_rate


def main():
    out_dir = pathlib.Path("samples")
    out_dir.mkdir(exist_ok=True, parents=True)

    y, sr = generate_demo_audio()
    sf.write(out_dir / "demo_gunshot_glass.wav", y, sr)

    y2, sr2 = generate_noise()
    sf.write(out_dir / "demo_noise.wav", y2, sr2)

    print(f"Generated samples in {out_dir}")


if __name__ == "__main__":
    main()
