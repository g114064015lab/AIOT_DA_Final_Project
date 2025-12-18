"""
Streamlit demo app for the Two-Stage Sound Event Detection (SED) and alert system.

This is a lightweight interactive front end aligned with the architecture in Arch.png:
PyAudio/ffmpeg (source) -> Librosa (pre-processing) -> Stage-1 CNN (edge) ->
Redis buffer (not used here) -> Transformer/CRNN (sequence refine) -> UI alerts.

The app uses a synthetic detector for demonstration; integrate your real Torch
models by replacing Stage1CNNEdgeDetector and Stage2SequenceRefiner.
"""

from __future__ import annotations

import io
import math
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import altair as alt
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
import pathlib


# --------- Data structures ----------------------------------------------------

@dataclass
class DetectionEvent:
    start: float
    end: float
    label: str
    score: float
    stage: str


# --------- Demo pipeline components ------------------------------------------

class Stage1CNNEdgeDetector:
    """
    Placeholder for the lightweight CNN edge detector.
    Uses energy + spectral centroid heuristics to mimic coarse detection.
    """

    def __init__(self, sample_rate: int, threshold: float):
        self.sample_rate = sample_rate
        self.threshold = threshold

    def predict(self, y: np.ndarray, frame_length: int, hop_length: int) -> List[DetectionEvent]:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, hop_length=hop_length
        )[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        cent_norm = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)
        energy_score = 0.6 * rms_norm + 0.4 * cent_norm

        frame_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=self.sample_rate, hop_length=hop_length
        )
        events: List[DetectionEvent] = []
        active = energy_score > self.threshold
        start_idx = None
        for idx, is_active in enumerate(active):
            if is_active and start_idx is None:
                start_idx = idx
            if not is_active and start_idx is not None:
                end_idx = idx
                events.append(
                    DetectionEvent(
                        start=float(frame_times[start_idx]),
                        end=float(frame_times[end_idx]),
                        label="candidate",
                        score=float(energy_score[start_idx:end_idx].max()),
                        stage="stage1",
                    )
                )
                start_idx = None
        if start_idx is not None:
            events.append(
                DetectionEvent(
                    start=float(frame_times[start_idx]),
                    end=float(frame_times[-1]),
                    label="candidate",
                    score=float(energy_score[start_idx:].max()),
                    stage="stage1",
                )
            )
        return events


class Stage2SequenceRefiner:
    """
    Placeholder for Transformer/CRNN sequence refinement.
    Applies smoothing and re-labeling to mimic temporal consistency.
    """

    def __init__(self, class_map: Iterable[str], min_duration: float, bonus: float):
        self.class_map = list(class_map)
        self.min_duration = min_duration
        self.bonus = bonus

    def refine(self, events: List[DetectionEvent]) -> List[DetectionEvent]:
        refined: List[DetectionEvent] = []
        for ev in events:
            duration = ev.end - ev.start
            if duration < self.min_duration:
                ev.end = ev.start + self.min_duration
            label_idx = int(math.floor(ev.score * len(self.class_map))) % len(self.class_map)
            label = self.class_map[label_idx]
            score = min(1.0, ev.score + self.bonus)
            refined.append(
                DetectionEvent(
                    start=ev.start,
                    end=ev.end,
                    label=label,
                    score=score,
                    stage="stage2",
                )
            )
        return refined


# --------- Helpers ------------------------------------------------------------

def load_audio(file: io.BytesIO, sample_rate: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(file, sr=sample_rate, mono=True)
    return y, sr


def generate_demo_audio(duration: float = 8.0, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
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


def plot_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Log-Mel Spectrogram")
    fig.tight_layout()
    return fig


def format_events(events: List[DetectionEvent]) -> List[dict]:
    return [
        {
            "Start (s)": round(ev.start, 2),
            "End (s)": round(ev.end, 2),
            "Label": ev.label,
            "Score": round(ev.score, 3),
            "Stage": ev.stage,
        }
        for ev in events
    ]


def events_to_df(events: List[DetectionEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": ev.start,
                "end": ev.end,
                "label": ev.label,
                "score": ev.score,
                "stage": ev.stage,
                "duration": max(ev.end - ev.start, 0.0001),
            }
            for ev in events
        ]
    )


def build_waterfall_spectrogram(
    y: np.ndarray,
    sr: int,
    overlay_events: List[DetectionEvent] | None = None,
) -> plt.Figure:
    """Waterfall-style log-mel spectrogram (red=高能?? ??低能?? with optional event overlays."""
    hop_length = 512
    n_mels = 64
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(S_dB.shape[1]), sr=sr, hop_length=hop_length)
    vmax = float(np.percentile(S_dB, 99))
    vmin = float(np.percentile(S_dB, 5))
    # 保??60 dB ??範?，避?全??
    min_db = max(vmin, vmax - 60.0)
    max_db = vmax

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(
        S_dB,
        origin="lower",
        aspect="auto",
        extent=[times.min(), times.max(), 0, n_mels],
        vmin=min_db,
        vmax=max_db,
        cmap="jet",  # red hot, blue cold
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Waterfall (Log-Mel, dB)", color="#e9edff")
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(0, n_mels)
    ax.set_xticks(np.linspace(times.min(), times.max(), 9))
    ax.set_yticks(np.linspace(0, n_mels, 9))
    ax.tick_params(colors="#cfd5ff")
    ax.grid(alpha=0.15, color="gray")
    cbar = fig.colorbar(img, ax=ax, label="dB")
    cbar.ax.tick_params(colors="#cfd5ff")
    cbar.set_label("dB", color="#cfd5ff")

    if overlay_events:
        for idx, ev in enumerate(overlay_events):
            ax.axvspan(ev.start, ev.end, color="white", alpha=0.12, linewidth=0)
            ax.text(
                (ev.start + ev.end) / 2,
                n_mels * 0.9 - (idx % 5) * (n_mels * 0.12),
                f"{ev.label}",
                ha="center",
                va="center",
                fontsize=9,
                color="#e9edff",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none"),
            )

    ax.set_facecolor("#0b0f1a")
    fig.patch.set_facecolor("#0b0f1a")
    fig.tight_layout()
    return fig


def build_event_pie(events: List[DetectionEvent]) -> plt.Figure:
    """Slide-style donut chart (matplotlib) with highlighted top share."""
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0b0f1a")
    ax.set_facecolor("#0b0f1a")

    if not events:
        ax.text(0.5, 0.5, "No events", ha="center", va="center", color="#e9edff", fontsize=14)
        ax.axis("off")
        return fig

    agg: dict[str, float] = {}
    for ev in events:
        agg[ev.label] = agg.get(ev.label, 0.0) + max(ev.end - ev.start, 0.0)

    labels = list(agg.keys())
    durations = np.array([agg[k] for k in labels], dtype=float)
    total = durations.sum()
    if total <= 0:
        durations = np.ones_like(durations)
        total = durations.sum()
    pct = durations / total * 100

    max_idx = int(np.argmax(durations))
    highlight_color = "#5b9bff"
    base_colors = ["#bfc4cc", "#d4d8df", "#8d939c", "#c8cdd5", "#9fa5ae", "#dde2ea"]
    colors = [highlight_color if i == max_idx else base_colors[i % len(base_colors)] for i in range(len(labels))]
    explode = [0.09 if i == max_idx else 0.03 for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        durations,
        labels=None,
        explode=explode,
        colors=colors,
        startangle=110,
        shadow=False,
        wedgeprops={"edgecolor": "#0b0f1a", "linewidth": 1.0},
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        pctdistance=0.8,
    )

    circle = plt.Circle((0, 0), 0.55, color="#0b0f1a")
    ax.add_artist(circle)

    ax.text(
        0,
        0.05,
        f"{pct[max_idx]:.0f}%",
        ha="center",
        va="center",
        fontsize=34,
        fontweight="bold",
        color=highlight_color,
    )
    ax.text(
        0,
        -0.12,
        labels[max_idx],
        ha="center",
        va="center",
        fontsize=12,
        color="#e9edff",
    )

    for i, (w, lab, p) in enumerate(zip(wedges, labels, pct)):
        ang = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        ha = "left" if x > 0 else "right"
        ax.text(
            1.2 * x,
            1.2 * y,
            f"{lab} ({p:.0f}%)",
            ha=ha,
            va="center",
            fontsize=10,
            color="#e9edff",
        )

    plt.setp(texts, size=0)
    plt.setp(autotexts, size=0)
    ax.set_title("Event Duration Share (Stage-2)", fontsize=15, color="#e9edff", pad=12)
    ax.axis("equal")
    return fig


def load_loudest_sample(samples_dir: pathlib.Path) -> Tuple[np.ndarray, int] | None:
    """Scan samples directory and return the loudest WAV by RMS."""
    wavs = sorted(samples_dir.glob("*.wav"))
    best = None
    best_rms = -1.0
    for wav in wavs:
        try:
            audio, sr = sf.read(wav)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            rms = np.sqrt(np.mean(audio**2))
            if rms > best_rms:
                best_rms = rms
                best = (audio.astype(np.float32), sr)
        except Exception:
            continue
    return best


def sample_geo_events() -> pd.DataFrame:
    # Mocked city coordinates (Taipei area) with class/score to visualize on a 3D map.
    data = [
        {"lat": 25.0330, "lon": 121.5654, "class": "gunshot", "score": 0.82},
        {"lat": 25.0478, "lon": 121.5319, "class": "glass_break", "score": 0.65},
        {"lat": 25.0418, "lon": 121.5080, "class": "siren", "score": 0.72},
        {"lat": 25.0520, "lon": 121.5430, "class": "scream", "score": 0.55},
        {"lat": 25.0260, "lon": 121.5270, "class": "gunshot", "score": 0.91},
        {"lat": 25.0600, "lon": 121.5200, "class": "glass_break", "score": 0.60},
        {"lat": 25.0300, "lon": 121.5500, "class": "siren", "score": 0.70},
        {"lat": 25.0570, "lon": 121.5650, "class": "scream", "score": 0.58},
    ]
    return pd.DataFrame(data)


def render_event_chips(events: List[DetectionEvent], top_k: int = 6) -> None:
    if not events:
        st.info("尚無事件，?上傳???使??例?)
        return
    top_events = sorted(events, key=lambda e: e.score, reverse=True)[:top_k]
    chip_html = []
    for ev in top_events:
        chip_html.append(
            f"""
            <div class="chip">
                <div class="chip-label">{ev.label}</div>
                <div class="chip-meta">Score {ev.score:.3f} · {ev.start:.2f}s ??{ev.end:.2f}s · {ev.stage}</div>
            </div>
            """
        )
    st.markdown(
        f"""<div class="chip-row">{''.join(chip_html)}</div>""",
        unsafe_allow_html=True,
    )


# --------- Streamlit UI -------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="GUARD | Urban Acoustic SED & Alert Demo",
        page_icon="?",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            background:
              linear-gradient(125deg, rgba(43,45,48,1) 0%, rgba(43,45,48,0.96) 55%, rgba(43,45,48,0.92) 100%),
              radial-gradient(120% 120% at 20% 20%, rgba(255,215,150,0.18), transparent 60%),
              radial-gradient(140% 90% at 80% 10%, rgba(255,210,120,0.12), transparent 55%),
              repeating-linear-gradient(110deg, rgba(255,215,150,0.10) 0, rgba(255,215,150,0.10) 2px, transparent 2px, transparent 12px),
              repeating-linear-gradient(200deg, rgba(255,195,110,0.08) 0, rgba(255,195,110,0.08) 1px, transparent 1px, transparent 18px);
            color: #e7ecff;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                170deg,
                rgba(255,210,66,0.97) 0%,
                rgba(255,210,66,0.94) 45%,
                rgba(255,210,66,0.88) 70%,
                rgba(140,105,35,0.88) 88%,
                rgba(222,213,200,0.94) 100%
            );
            border-right: 1px solid rgba(255,215,170,0.35);
            box-shadow: inset -6px 0 12px rgba(0,0,0,0.15);
        }
        section[data-testid="stSidebar"] * {
            color: #3d2420 !important;
        }
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #3d2420 !important;
            font-weight: 800;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #fff2e8 !important;
            color: #2a1610 !important;
            border: 1px solid #d9a87f !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
        section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
            background: #fff2e8 !important;
            color: #2a1610 !important;
            border: 1px solid #d9a87f !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="track"] {
            background: #f5e2cf !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
            background: #e58c6f !important;
            border: 1px solid #c56a4d !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="tick"] {
            background: #8a5d4a !important;
        }
        section[data-testid="stSidebar"] .stNumberInput input {
            color: #2a1610 !important;
            background: #fff2e8 !important;
            border: 1px solid #d9a87f !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] .stNumberInput button {
            background: #1f1616 !important;
            color: #f1c9ad !important;
            border: 1px solid #d9a87f !important;
        }
        section[data-testid="stSidebar"] .stNumberInput svg {
            fill: #f1c9ad !important;
        }
        .sidebar-logo {
            margin-top: 18px;
            padding: 14px 12px 10px 12px;
            background: linear-gradient(135deg, rgba(255,210,66,0.28), rgba(222,213,200,0.42));
            border: 1px solid rgba(181,141,84,0.4);
            border-radius: 18px;
            text-align: center;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 10px 20px rgba(0,0,0,0.08);
        }
        .sidebar-logo .logo-icon {
            transform: scale(0.85);
            margin: 0 auto 4px auto;
        }
        .logo-text-mini {
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.6px;
            color: #3d2420;
            margin-bottom: 2px;
        }
        .logo-sub-mini {
            font-size: 11px;
            letter-spacing: 0.4px;
            color: #5a382b;
            opacity: 0.85;
        }
        .hero {
            background:
              linear-gradient(140deg, rgba(48,36,20,0.94) 0%, rgba(26,18,10,0.92) 35%, rgba(255,210,130,0.42) 100%),
              radial-gradient(180% 140% at 80% 0%, rgba(255,215,150,0.26) 0%, transparent 55%),
              repeating-linear-gradient(105deg, rgba(255,225,160,0.18) 0, rgba(255,225,160,0.18) 1px, transparent 1px, transparent 12px);
            padding: 28px 32px;
            border-radius: 18px;
            color: #f6f8ff;
            box-shadow: 0 24px 60px rgba(0,0,0,0.45);
            border: 1px solid rgba(255,215,170,0.22);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content:"";
            position:absolute;
            inset:0;
            background: repeating-linear-gradient(95deg, rgba(255,220,140,0.12) 0, rgba(255,220,140,0.12) 1px, transparent 1px, transparent 14px);
            opacity:0.35;
            pointer-events:none;
        }
        .logo-wrap {
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:10px;
        }
        .logo-icon {
            width:54px;
            height:54px;
            border-radius:16px;
            background: radial-gradient(circle at 30% 30%, rgba(255,230,170,0.95), rgba(255,195,110,0.75) 50%, rgba(30,18,10,0.9));
            box-shadow: 0 12px 28px rgba(0,0,0,0.5), inset 0 1px 8px rgba(255,255,255,0.4);
            display:flex;
            align-items:center;
            justify-content:center;
            color:#0b0f1a;
            position: relative;
            overflow: hidden;
        }
        .logo-letter {
            font-weight:900;
            font-size:30px;
            letter-spacing:-0.5px;
            color: rgb(128, 99, 88);
            font-family: 'Kunstler Script', cursive;
            z-index:2;
        }
        .logo-bars {
            position:absolute;
            right:6px;
            bottom:8px;
            display:flex;
            align-items:flex-end;
            gap:2px;
            z-index:1;
        }
        .logo-bars span {
            display:block;
            width:4px;
            background: linear-gradient(180deg, rgba(216,171,184,0.9), rgba(255,200,120,0.6));
            border-radius:2px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }
        .logo-bars span:nth-child(1) { height:10px; }
        .logo-bars span:nth-child(2) { height:16px; }
        .logo-bars span:nth-child(3) { height:8px; }
        .logo-bars span:nth-child(4) { height:14px; }
        }
        .logo-text {
            font-size:22px;
            font-weight:800;
            color:#f6f8ff;
            letter-spacing:0.3px;
        }
        .badge {
            display:inline-block;
            padding:6px 12px;
            border-radius:999px;
            background: rgba(255,255,255,0.85);
            color:#0c1530;
            font-weight:700;
            letter-spacing:0.25px;
        }
        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }
        /* Nav bar styling */
        div[data-testid="stRadio"] > div {
            flex-direction: row;
            gap: 16px;
            justify-content: flex-start;
            align-items: center;
        }
        div[data-testid="stRadio"] label {
            background: transparent;
            color: #e8ecff;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid transparent;
            transition: all 160ms ease;
            font-weight: 600;
            font-size: 16px;
        }
        div[data-testid="stRadio"] label:hover {
            color: #b8c7ff;
            border-color: rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            transform: translateY(-1px);
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            color: #a5baff;
            border-color: rgba(255,255,255,0.18);
            background: rgba(75,109,255,0.12);
        }
        .chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin: 8px 0 12px 0;
        }
        .chip {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 10px 12px;
            min-width: 180px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
            transition: all 200ms ease;
        }
        .chip:hover {
            transform: translateY(-2px);
            border-color: rgba(255,255,255,0.22);
        }
        .chip-label {
            font-weight:700;
            font-size:14px;
        }
        .chip-meta {
            font-size:12px;
            opacity:0.8;
        }
        .pill {
            display:inline-block;
            padding:6px 10px;
            border-radius:12px;
            background: rgba(67,111,255,0.18);
            border:1px solid rgba(67,111,255,0.35);
            color:#b8cbff;
            font-size:12px;
            margin-right:6px;
        }
        .card-3d {
            background: linear-gradient(145deg, rgba(30,32,45,0.95), rgba(18,18,26,0.9));
            border: 1px solid rgba(255,215,170,0.2);
            border-radius: 18px;
            box-shadow:
              0 20px 50px rgba(0,0,0,0.35),
              inset 0 1px 8px rgba(255,255,255,0.1);
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
        }
        .card-3d::before {
            content:"";
            position:absolute;
            inset: -30% 40% auto -30%;
            height: 120%;
            background: radial-gradient(circle at 30% 30%, rgba(255,215,150,0.16), transparent 55%);
            opacity: 0.6;
            pointer-events:none;
        }
        .card-3d h4 {
            margin: 0 0 6px 0;
            color: #f6f8ff;
        }
        .card-3d p {
            margin: 4px 0;
            color: #cfd5ff;
            font-size: 13px;
        }
        .card-3d small {
            color: #ffdd99;
            letter-spacing: 0.5px;
        }
        .manual-card {
            background: linear-gradient(135deg, rgba(20,22,30,0.95), rgba(14,10,6,0.9));
            border: 1px solid rgba(255,215,170,0.25);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
            margin-bottom: 10px;
        }
        .manual-step {
            display:flex;
            align-items:flex-start;
            gap:12px;
        }
        .manual-step-number {
            min-width:34px;
            height:34px;
            border-radius:10px;
            background: linear-gradient(135deg, #ffcf8f, #f0a950);
            color:#0b0f1a;
            font-weight:800;
            display:flex;
            align-items:center;
            justify-content:center;
            box-shadow: inset 0 1px 4px rgba(255,255,255,0.4);
        }
        .manual-step-text h4 {
            margin:0 0 4px 0;
            color:#f6f8ff;
        }
        .manual-step-text p {
            margin:0;
            color:#cfd5ff;
            font-size:13px;
        }
        .card-3d {
            background: linear-gradient(145deg, rgba(30,32,45,0.95), rgba(18,18,26,0.9));
            border: 1px solid rgba(255,215,170,0.2);
            border-radius: 18px;
            box-shadow:
              0 20px 50px rgba(0,0,0,0.35),
              inset 0 1px 8px rgba(255,255,255,0.1);
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
        }
        .card-3d::before {
            content:"";
            position:absolute;
            inset: -30% 40% auto -30%;
            height: 120%;
            background: radial-gradient(circle at 30% 30%, rgba(255,215,150,0.16), transparent 55%);
            opacity: 0.6;
            pointer-events:none;
        }
        .card-3d h4 {
            margin: 0 0 6px 0;
            color: #f6f8ff;
        }
        .card-3d p {
            margin: 4px 0;
            color: #cfd5ff;
            font-size: 13px;
        }
        .card-3d small {
            color: #ffdd99;
            letter-spacing: 0.5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="hero">
      <div class="logo-wrap">
        <div class="logo-icon">
          <span class="logo-letter">G</span>
          <div class="logo-bars">
            <span></span><span></span><span></span><span></span>
          </div>
        </div>
        <div class="logo-text">GUARD · General Urban Audio Recognition & Defense</div>
      </div>
      <h2 style="margin:12px 0 6px 0;font-size:32px;font-weight:800;">???音事件?測?公???警??/h2>
      <p style="margin:0;font-size:16px;font-weight:600;">GUARD: The City Never Sleeps, Neither Do We.</p>
      <p style="margin:6px 0 0 0;font-size:14px;opacity:0.9;">Two-Stage SED (CNN ??Transformer/CRNN) with Librosa preprocessing. Upload ?使????例音訊?調整?值???設?，查?偵測???/p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tabs = ["Products", "Applications", "Case Studies", "Support", "Architecture"]
    nav = st.radio("導航", tabs, horizontal=True, label_visibility="collapsed", key="nav_radio")

    if nav != "Products":
        st.sidebar.info("?? Products ?即?使???、推論?????)

    with st.sidebar:
        st.header("?? ??設?")
        sr = st.number_input("Sample rate", value=16000, step=1000, min_value=8000, max_value=48000)
        frame_len_sec = st.slider("Frame length (seconds)", 0.5, 2.5, 1.0, 0.25)
        hop_len_sec = st.slider("Hop length (seconds)", 0.1, 1.0, 0.25, 0.05)
        stage1_threshold = st.slider("Stage-1 energy threshold", 0.05, 0.9, 0.35, 0.01)
        stage2_bonus = st.slider("Stage-2 score bonus", 0.0, 0.5, 0.1, 0.01)
        min_duration = st.slider("Stage-2 min duration (s)", 0.1, 2.0, 0.4, 0.1)
        class_map = st.multiselect(
            "事件類別?? (示?)",
            options=["gunshot", "glass_break", "car_horn", "scream", "other"],
            default=["gunshot", "glass_break", "scream"],
        )
        st.divider()
        st.markdown("**互???**")
        show_spectrogram = st.checkbox("顯示????, value=True)
        allow_download = st.checkbox("?許下??測結? CSV", value=True)
        st.markdown(
            """
            <div class=\"sidebar-logo\">
              <div class=\"logo-icon\">
                <span class=\"logo-letter\">G</span>
                <div class=\"logo-bars\"><span></span><span></span><span></span><span></span></div>
              </div>
              <div class=\"logo-text-mini\">GUARD · SED</div>
              <div class=\"logo-sub-mini\">The City Never Sleeps</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if nav == "Products":
        st.subheader("1) 載入??")
        st.markdown('<span class="pill">Upload</span><span class="pill">Demo</span><span class="pill">Adjust Thresholds</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("上傳 WAV/OGG/FLAC/MP3", type=["wav", "ogg", "flac", "mp3"])
        use_demo = st.checkbox("使用?建??範???（含槍響+?????, value=uploaded is None)
        use_loudest = st.checkbox("?用 samples/ 中?????0 ??sample_*?, value=False)
        sample_choices = []
        samples_dir = pathlib.Path("samples")
        if samples_dir.exists():
            sample_choices = sorted([p.name for p in samples_dir.glob("sample_*.wav")])
        chosen_sample = None
        if sample_choices:
            chosen_sample = st.selectbox("?選????sample_* 檔??放", options=["(不選)"] + sample_choices, index=0)
        audio_bytes: bytes | None = None
        audio_np: np.ndarray | None = None

        if uploaded is not None:
            audio_bytes = uploaded.read()
            audio_np, sr = load_audio(io.BytesIO(audio_bytes), sample_rate=sr)
        elif chosen_sample and chosen_sample != "(不選)":
            try:
                path = samples_dir / chosen_sample
                audio_np, sr = sf.read(path)
                if audio_np.ndim > 1:
                    audio_np = np.mean(audio_np, axis=1)
                audio_np = audio_np.astype(np.float32)
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sr, format="WAV")
                audio_bytes = buffer.getvalue()
            except Exception:
                audio_np = None
        elif use_loudest:
            best = load_loudest_sample(pathlib.Path("samples"))
            if best is not None:
                audio_np, sr = best
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sr, format="WAV")
                audio_bytes = buffer.getvalue()
        elif use_demo:
            audio_np, sr = generate_demo_audio(sample_rate=sr)
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, sr, format="WAV")
            audio_bytes = buffer.getvalue()

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

        if audio_np is None:
            st.info("請??音訊??用??範???)
            return

        st.subheader("2) ?徵?兩?段??")
        frame_length = int(frame_len_sec * sr)
        hop_length = int(hop_len_sec * sr)

        stage1 = Stage1CNNEdgeDetector(sample_rate=sr, threshold=stage1_threshold)
        stage2 = Stage2SequenceRefiner(class_map=class_map or ["other"], min_duration=min_duration, bonus=stage2_bonus)

        with st.spinner("???測中?):
            stage1_events = stage1.predict(audio_np, frame_length=frame_length, hop_length=hop_length)
            refined_events = stage2.refine(stage1_events)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**?段 1：CNN ?緣?測 (示?)**")
            st.dataframe(format_events(stage1_events), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**?段 2：Transformer/CRNN ??精? (示?)**")
            st.dataframe(format_events(refined_events), use_container_width=True, hide_index=True)

        # Metrics summary cards
        total_stage1 = len(stage1_events)
        total_stage2 = len(refined_events)
        max_score = max([ev.score for ev in refined_events], default=0.0)
        unique_labels = len({ev.label for ev in refined_events}) if refined_events else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Stage-1 事件??, total_stage1)
        m2.metric("Stage-2 事件??, total_stage2)
        m3.metric("?高置信度", f"{max_score:.3f}", help="?Stage-2 平?後???score")
        st.markdown("**Top Events (?調?顯示數??**")
        top_k = st.slider("顯示??N ?, 3, 12, 6, 1)

        if allow_download and refined_events:
            csv_buffer = io.StringIO()
            csv_buffer.write("id,start,end,label,score,stage\n")
            for ev in refined_events:
                csv_buffer.write(
                    f"{uuid.uuid4().hex},{ev.start:.3f},{ev.end:.3f},{ev.label},{ev.score:.3f},{ev.stage}\n"
                )
            st.download_button(
                "下??測結? CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="sed_events.csv",
                mime="text/csv",
            )

        st.subheader("3) 視覺??互?")
        if show_spectrogram:
            pie_fig = build_event_pie(refined_events)
            if isinstance(pie_fig, (alt.Chart, alt.LayerChart, alt.ConcatChart, alt.HConcatChart, alt.VConcatChart, alt.FacetChart, alt.RepeatChart, alt.TopLevelMixin)):
                st.altair_chart(pie_fig, use_container_width=True)
            else:
                st.pyplot(pie_fig, clear_figure=True, use_container_width=True)

        st.markdown("**事件???(Stage1 / Stage2)**")
        df_events = events_to_df(stage1_events + refined_events)
        if not df_events.empty:
            stages = df_events["stage"].unique().tolist()
            labels = df_events["label"].unique().tolist()
            stage_filter = st.multiselect("篩選?段", options=stages, default=stages)
            label_filter = st.multiselect("篩選類別", options=labels, default=labels)
            filtered = df_events[
                df_events["stage"].isin(stage_filter) & df_events["label"].isin(label_filter)
            ]
            if not filtered.empty:
                chart = (
                    alt.Chart(filtered)
                    .mark_bar(cornerRadius=6)
                    .encode(
                        x=alt.X("start:Q", title="Start (s)"),
                        x2="end:Q",
                        y=alt.Y("label:N", title="Label"),
                        color=alt.Color("stage:N", scale=alt.Scale(scheme="tableau20")),
                        tooltip=[
                            alt.Tooltip("label:N", title="Label"),
                            alt.Tooltip("stage:N", title="Stage"),
                            alt.Tooltip("start:Q", title="Start (s)", format=".2f"),
                            alt.Tooltip("end:Q", title="End (s)", format=".2f"),
                            alt.Tooltip("score:Q", title="Score", format=".3f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("?符?篩??件?事件??)

        st.subheader("4) 如????實模??)
        st.markdown(
            """
        - ?TorchScript ??ONNX 載入你? Stage-1 CNN，? `Stage1CNNEdgeDetector.predict` ?為模?????
        - ?Stage-2 ??已?練? Transformer/CRNN，輸???特徵? logits，輸??件?表?
        - ?? Redis 緩?，? Stage-1 ????logits/?徵?入緩?，???Stage-2 ?次讀??
        - 將?警管??Webhook/SMS/Email）接??Stage-2 結?上?依??值??卻???送?
        """
        )
    elif nav == "Applications":
        st.subheader("??安全???場??· 3D Map 互?")
        st.markdown(
            """
            ?地????事件類??強度??解 GUARD 如???市中布署??
            - 事件越亮/??越??置信度?高?
            - ?篩???、調?柱高?例?????
            - ???實佈?：以實? lat/lon ??class/score ?? `sample_geo_events()`.
            """
        )
        geo_df = sample_geo_events()
        classes = sorted(geo_df["class"].unique().tolist())
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sel_classes = st.multiselect("顯示類別", classes, default=classes)
        with col_ctrl2:
            radius = st.slider("?? (?尺)", 80, 300, 160, 10)
        with col_ctrl3:
            height_scale = st.slider("高度比?", 50, 300, 120, 10)

        filtered = geo_df[geo_df["class"].isin(sel_classes)].copy()
        # Map color by class
        color_map = {
            "gunshot": [255, 90, 90],
            "glass_break": [255, 180, 80],
            "siren": [90, 180, 255],
            "scream": [160, 120, 255],
        }
        filtered["color"] = filtered["class"].apply(lambda c: color_map.get(c, [200, 200, 200]))
        filtered["elevation"] = (filtered["score"] * height_scale).astype(float)

        midpoint = [filtered["lat"].mean(), filtered["lon"].mean()] if not filtered.empty else [25.04, 121.56]
        layer = pdk.Layer(
            "ColumnLayer",
            data=filtered,
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=radius,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            longitude=midpoint[1],
            latitude=midpoint[0],
            zoom=12.5,
            min_zoom=5,
            max_zoom=18,
            pitch=45,
            bearing=15,
        )
        tooltip = {"text": "Class: {class}\nScore: {score}\nLat: {lat}\nLon: {lon}"}
        map_col, legend_col = st.columns([4, 1])
        with map_col:
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/dark-v11"))
        with legend_col:
            st.markdown("**??**")
            for lbl, col in color_map.items():
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
                    f"<span style='display:inline-block;width:14px;height:14px;border-radius:4px;background:rgb({col[0]},{col[1]},{col[2]});'></span>"
                    f"<span style='color:#e9edff;font-size:12px;'>{lbl}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**交通????維**")
            st.write(
                "- ??路口警?辨?，信?優???\n"
                "- ??/?鐵站???尖叫/求?觸發安?\n"
                "- 工地/?工????常???即?通報"
            )
        with c2:
            st.markdown("**???護????*")
            st.write(
                "- ??/?院/?場?玻?破裂??入?測\n"
                "- 社?夜?巡防：異常??聲?爭???\n"
                "- ?慧建?：??異?、設?異常噪???
            )
    elif nav == "Case Studies":
        st.subheader("案?示?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="card-3d">
                  <small>?慧街? · 3D</small>
                  <h4>多?麥?風陣??/h4>
                  <p>Stage-1 高召??Stage-2 ?誤??誤報??< 2%??/p>
                  <p>串接 Redis 緩???件????援??????/p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="card-3d">
                  <small>???護 · 立??場</small>
                  <h4>???? / 尖叫??</h4>
                  <p>事件?警??< 2 秒?夜??音?適??/p>
                  <p>?搭??CCTV/?禁??? 3D ?????放??/p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="card-3d">
                  <small>交通??· 3D ??</small>
                  <h4>警? / ?? / 車?</h4>
                  <p>?信?優?串????事件縮???3D ?景??/p>
                  <p>低延??API，誤????多??離??/p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif nav == "Support":
        st.subheader("?援?部?)
        st.markdown(
            """
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">1</div>
                <div class="manual-step-text">
                  <h4>??準?</h4>
                  <p>安? Python 3.11，`pip install -r requirements.txt`，確?ffmpeg ??PATH??/p>
                </div>
              </div>
            </div>
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">2</div>
                <div class="manual-step-text">
                  <h4>????</h4>
                  <p>??案根????：`streamlit run app.py`?雲端部署??用 headless??/p>
                </div>
              </div>
            </div>
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">3</div>
                <div class="manual-step-text">
                  <h4>載入?推?/h4>
                  <p>上傳???選??samples，調??frame/hop?閾??檢?事件???CSV??/p>
                </div>
              </div>
            </div>
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">4</div>
                <div class="manual-step-text">
                  <h4>??模?</h4>
                  <p>??`requirements-train.txt` 訓練 Stage-1/2，???TorchScript/ONNX，替??`Stage1CNNEdgeDetector` / `Stage2SequenceRefiner`??/p>
                </div>
              </div>
            </div>
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">5</div>
                <div class="manual-step-text">
                  <h4>?署????/h4>
                  <p>???? + TLS；Webhook 權?/簽?；Redis 設??控??????/p>
                </div>
              </div>
            </div>
            <div class="manual-card">
              <div class="manual-step">
                <div class="manual-step-number">6</div>
                <div class="manual-step-text">
                  <h4>?難?除</h4>
                  <p>torch 安?失?：改??Python 3.10??.12?ffmpeg 缺失：?裝???終端??/p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  # Architecture
        st.subheader("Architecture · GUARD 端到端??)
        st.markdown(
            """
            - PyAudio/ffmpeg：接??市音訊?
            - Librosa：特徵抽?、???Log-Mel/PCEN）?
            - Stage-1 CNN：?段?高召?偵測?
            - Redis：特?logits 緩?供??模????
            - Transformer/序?模?：?序精??誤報??
            - Deployment/Inference Service：輸??件、警?? API??
            """
        )
        svg_arch = """
<svg width="120%" height="500" viewBox="0 0 1180 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4ea1ff"/>
      <stop offset="100%" stop-color="#7dd8ff"/>
    </linearGradient>
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ff9f71"/>
      <stop offset="100%" stop-color="#ffcba2"/>
    </linearGradient>
    <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="12" flood-color="rgba(0,0,0,0.35)"/>
    </filter>
    <marker id="arrowHead" markerWidth="10" markerHeight="8" refX="5" refY="4" orient="auto">
      <polygon points="0 0, 10 4, 0 8" fill="rgba(255,255,255,0.85)" />
    </marker>
  </defs>
  <style>
    .card { fill: rgba(255,255,255,0.06); stroke: rgba(255,255,255,0.22); stroke-width:1.6; rx:16; ry:16; filter:url(#cardShadow);}
    .title { fill: #e9edff; font-size: 18px; font-family: 'Space Grotesk','Segoe UI',sans-serif; font-weight:700; }
    .desc { fill: #cdd5ff; font-size: 14px; font-family: 'Space Grotesk','Segoe UI',sans-serif; }
    .badgeBlue { fill: url(#blueGrad); }
    .badgeOrange { fill: url(#orangeGrad); }
    .arrow { stroke: rgba(255,255,255,0.65); stroke-width:2; marker-end: url(#arrowHead); }
  </style>

  <!-- Arrows beneath nodes -->
  <line x1="210" y1="210" x2="255" y2="120" class="arrow"/>
  <line x1="210" y1="210" x2="255" y2="275" class="arrow"/>
  <line x1="460" y1="125" x2="515" y2="185" class="arrow"/>
  <line x1="460" y1="280" x2="515" y2="200" class="arrow"/>
  <line x1="720" y1="205" x2="775" y2="120" class="arrow"/>
  <line x1="720" y1="205" x2="775" y2="275" class="arrow"/>

  <!-- Nodes on top -->
  <g>
  <!-- Source -->
  <rect x="40" y="150" width="170" height="120" class="card"/>
  <rect x="60" y="165" width="46" height="20" rx="10" class="badgeBlue"/>
  <text x="60" y="205" class="title">?? PyAudio</text>
  <text x="60" y="230" class="desc">ffmpeg capture</text>

  <!-- Librosa -->
  <rect x="260" y="60" width="200" height="100" class="card"/>
  <rect x="280" y="85" width="46" height="20" rx="10" class="badgeBlue"/>
  <text x="280" y="125" class="title">? Librosa</text>
  <text x="280" y="150" class="desc">Feature extract / slice</text>

  <!-- Stage1 CNN -->
  <rect x="260" y="220" width="200" height="100" class="card"/>
  <rect x="280" y="235" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="280" y="275" class="title">? Stage-1 CNN</text>
  <text x="280" y="300" class="desc">Segment edge detect</text>

  <!-- Redis -->
  <rect x="520" y="135" width="200" height="120" class="card"/>
  <rect x="540" y="160" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="540" y="200" class="title">?? Redis Buffer</text>
  <text x="540" y="225" class="desc">Feature/logits cache</text>

  <!-- Transformer -->
  <rect x="780" y="60" width="200" height="100" class="card"/>
  <rect x="800" y="85" width="46" height="20" rx="10" class="badgeBlue"/>
  <text x="800" y="125" class="title">?? Transformer</text>
  <text x="800" y="150" class="desc">Sequence refine</text>

  <!-- Inference -->
  <rect x="780" y="220" width="200" height="100" class="card"/>
  <rect x="800" y="235" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="800" y="275" class="title">?? Inference Service</text>
  <text x="800" y="300" class="desc">Alerts / API</text>
  </g>
</svg>
"""
        st.markdown(svg_arch, unsafe_allow_html=True)
        st.caption("GUARD Pipeline ????點顯示?源、????兩?段模??緩衝???")


if __name__ == "__main__":
    main()

