"""
Generate many sample audio clips for quick testing of the Streamlit demo.
Defaults to produce 50 short variants to keep repo size manageable.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import soundfile as sf


def gunshot_glass(duration: float, sample_rate: int, rng: np.random.Generator):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise = rng.normal(0, 0.02, size=t.shape)
    gunshot_center = int(rng.uniform(0.2, 0.8) * len(t))
    gunshot = np.zeros_like(t)
    gunshot[gunshot_center : gunshot_center + 200] = 1.0
    gunshot = np.convolve(gunshot, np.hanning(100), mode="same")
    glass_center = int(rng.uniform(0.3, 0.9) * len(t))
    glass = np.zeros_like(t)
    length = 400
    glass[glass_center : glass_center + length] = (
        np.sin(2 * np.pi * rng.uniform(3000, 6500) * t[:length]) * np.hanning(length)
    )
    y = noise + 0.9 * gunshot + 0.5 * glass
    y = y / np.abs(y).max()
    return y.astype(np.float32)


def siren(duration: float, sample_rate: int, rng: np.random.Generator):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    base = 0.05 * rng.normal(0, 1, size=t.shape)
    sweep = 0.5 * np.sin(2 * np.pi * (2 + 1.5 * np.sin(2 * np.pi * 0.3 * t)) * t)
    tone = 0.2 * np.sin(2 * np.pi * rng.uniform(400, 900) * t)
    y = base + sweep + tone
    y = y / np.abs(y).max()
    return y.astype(np.float32)


def noise(duration: float, sample_rate: int, rng: np.random.Generator):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    white = rng.normal(0, 0.06, size=t.shape)
    hum = 0.08 * np.sin(2 * np.pi * 60 * t)
    y = white + hum
    y = y / np.abs(y).max()
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per sample")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    out_dir = pathlib.Path("samples")
    out_dir.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(0xC0DEFACE)

    patterns = [gunshot_glass, siren, noise]
    pattern_names = ["gunshot_glass", "siren", "noise"]

    for i in range(args.count):
        pattern = patterns[i % len(patterns)]
        name = pattern_names[i % len(pattern_names)]
        y = pattern(args.duration, args.sr, rng)
        sf.write(out_dir / f"sample_{i:02d}_{name}.wav", y, args.sr)

    print(f"Generated {args.count} samples in {out_dir}")


if __name__ == "__main__":
    main()
