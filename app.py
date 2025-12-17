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
    """Waterfall-style log-mel spectrogram (red=高能量, 藍=低能量) with optional event overlays."""
    hop_length = 512
    n_mels = 64
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(S_dB.shape[1]), sr=sr, hop_length=hop_length)
    vmax = float(np.percentile(S_dB, 99))
    vmin = float(np.percentile(S_dB, 5))
    # 保持約 60 dB 動態範圍，避免全藍
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
    """Interactive donut (Altair) with hover highlight and center callout."""
    if not events:
        return (
            alt.Chart(pd.DataFrame({"text": ["No events"]}))
            .mark_text(size=16, color="#e9edff")
            .encode(text="text")
            .properties(width=320, height=220)
            .configure_view(stroke=None)
        )

    agg: dict[str, float] = {}
    for ev in events:
        agg[ev.label] = agg.get(ev.label, 0.0) + max(ev.end - ev.start, 0.0)

    df = pd.DataFrame(
        [
            {"label": k, "duration": v, "pct": v / sum(agg.values()) * 100.0}
            for k, v in agg.items()
        ]
    )
    top = df.loc[df["duration"].idxmax()]

    sel = alt.selection_single(fields=["label"], empty="none", on="mouseover")
    base = (
        alt.Chart(df)
        .encode(
            theta=alt.Theta("duration:Q", stack=True),
            color=alt.Color("label:N", legend=None, scale=alt.Scale(scheme="tableau20")),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("duration:Q", title="Duration (s)", format=".2f"),
                alt.Tooltip("pct:Q", title="Share (%)", format=".1f"),
            ],
        )
        .add_selection(sel)
    )

    wedges = base.mark_arc(innerRadius=80, stroke="white", strokeWidth=1.2).encode(
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.6))
    )
    labels = base.mark_text(radius=180, fontSize=10, color="#e9edff").encode(
        text=alt.Text("label:N"),
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.5)),
    )
    center = (
        alt.Chart(pd.DataFrame({"text": [f"{top['pct']:.0f}%", top["label"]]}))
        .mark_text(color="#e9edff")
        .encode(
            text="text:N",
            y=alt.Y("row_number():O", axis=None),
        )
        .transform_window(row_number="count()")
        .properties(width=0, height=0)
    )

    chart = (wedges + labels).properties(width=460, height=460, title="Event Duration Share (Stage-2)")
    chart = (
        chart.configure_view(stroke=None)
        .configure_title(color="#e9edff", fontSize=16)
        .configure_axis(labelColor="#e9edff", titleColor="#e9edff")
    )
    return chart


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
        st.info("尚無事件，請上傳音訊或使用範例。")
        return
    top_events = sorted(events, key=lambda e: e.score, reverse=True)[:top_k]
    chip_html = []
    for ev in top_events:
        chip_html.append(
            f"""
            <div class="chip">
                <div class="chip-label">{ev.label}</div>
                <div class="chip-meta">Score {ev.score:.3f} · {ev.start:.2f}s → {ev.end:.2f}s · {ev.stage}</div>
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
        page_icon="🎧",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
            background:
              radial-gradient(140% 140% at 15% 25%, rgba(20,20,28,0.95) 0%, rgba(8,8,12,0.96) 45%, #030508 80%),
              repeating-linear-gradient(110deg, rgba(255,215,150,0.16) 0, rgba(255,215,150,0.16) 2px, transparent 2px, transparent 16px),
              repeating-linear-gradient(200deg, rgba(255,195,110,0.10) 0, rgba(255,195,110,0.10) 1px, transparent 1px, transparent 22px);
            color: #e7ecff;
        }
        .hero {
            background:
              linear-gradient(140deg, rgba(32,32,44,0.92) 0%, rgba(18,12,6,0.9) 35%, rgba(240,190,110,0.22) 100%),
              radial-gradient(180% 140% at 80% 0%, rgba(255,215,150,0.12) 0%, transparent 45%);
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="hero">
      <div class="badge" style="padding:10px 16px;font-size:15px;">GUARD · General Urban Audio Recognition & Defense</div>
      <h2 style="margin:14px 0 8px 0;font-size:32px;font-weight:800;">城市聲音事件偵測與公共安全警報</h2>
      <p style="margin:0;font-size:16px;font-weight:600;">GUARD: The City Never Sleeps, Neither Do We.</p>
      <p style="margin:6px 0 0 0;font-size:14px;opacity:0.9;">Two-Stage SED (CNN → Transformer/CRNN) with Librosa preprocessing. Upload 或使用合成範例音訊，調整閾值與時序設定，查看偵測結果。</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tabs = ["Products", "Applications", "Case Studies", "Support", "Architecture"]
    nav = st.radio("導航", tabs, horizontal=True, label_visibility="collapsed", key="nav_radio")

    if nav != "Products":
        st.sidebar.info("切回 Products 頁即可使用上傳、推論與可視化。")

    with st.sidebar:
        st.header("⚙️ 推論設定")
        sr = st.number_input("Sample rate", value=16000, step=1000, min_value=8000, max_value=48000)
        frame_len_sec = st.slider("Frame length (seconds)", 0.5, 2.5, 1.0, 0.25)
        hop_len_sec = st.slider("Hop length (seconds)", 0.1, 1.0, 0.25, 0.05)
        stage1_threshold = st.slider("Stage-1 energy threshold", 0.05, 0.9, 0.35, 0.01)
        stage2_bonus = st.slider("Stage-2 score bonus", 0.0, 0.5, 0.1, 0.01)
        min_duration = st.slider("Stage-2 min duration (s)", 0.1, 2.0, 0.4, 0.1)
        class_map = st.multiselect(
            "事件類別映射 (示意)",
            options=["gunshot", "glass_break", "car_horn", "scream", "other"],
            default=["gunshot", "glass_break", "scream"],
        )
        st.divider()
        st.markdown("**互動元素**")
        show_spectrogram = st.checkbox("顯示頻譜圖", value=True)
        allow_download = st.checkbox("允許下載偵測結果 CSV", value=True)

    if nav == "Products":
        st.subheader("1) 載入音訊")
        st.markdown('<span class="pill">Upload</span><span class="pill">Demo</span><span class="pill">Adjust Thresholds</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("上傳 WAV/OGG/FLAC/MP3", type=["wav", "ogg", "flac", "mp3"])
        use_demo = st.checkbox("使用內建合成範例音訊（含槍響+玻璃破裂）", value=uploaded is None)
        use_loudest = st.checkbox("改用 samples/ 中最響的樣本（50 個 sample_*）", value=False)
        sample_choices = []
        samples_dir = pathlib.Path("samples")
        if samples_dir.exists():
            sample_choices = sorted([p.name for p in samples_dir.glob("sample_*.wav")])
        chosen_sample = None
        if sample_choices:
            chosen_sample = st.selectbox("或選擇一個 sample_* 檔案播放", options=["(不選)"] + sample_choices, index=0)
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
            st.info("請上傳音訊或啟用合成範例。")
            return

        st.subheader("2) 特徵與兩階段推論")
        frame_length = int(frame_len_sec * sr)
        hop_length = int(hop_len_sec * sr)

        stage1 = Stage1CNNEdgeDetector(sample_rate=sr, threshold=stage1_threshold)
        stage2 = Stage2SequenceRefiner(class_map=class_map or ["other"], min_duration=min_duration, bonus=stage2_bonus)

        with st.spinner("運行偵測中…"):
            stage1_events = stage1.predict(audio_np, frame_length=frame_length, hop_length=hop_length)
            refined_events = stage2.refine(stage1_events)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**階段 1：CNN 邊緣偵測 (示意)**")
            st.dataframe(format_events(stage1_events), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**階段 2：Transformer/CRNN 時序精煉 (示意)**")
            st.dataframe(format_events(refined_events), use_container_width=True, hide_index=True)

        # Metrics summary cards
        total_stage1 = len(stage1_events)
        total_stage2 = len(refined_events)
        max_score = max([ev.score for ev in refined_events], default=0.0)
        unique_labels = len({ev.label for ev in refined_events}) if refined_events else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Stage-1 事件數", total_stage1)
        m2.metric("Stage-2 事件數", total_stage2)
        m3.metric("最高置信度", f"{max_score:.3f}", help="經 Stage-2 平滑後的最大 score")
        st.markdown("**Top Events (可調整顯示數量)**")
        top_k = st.slider("顯示前 N 筆", 3, 12, 6, 1)

        if allow_download and refined_events:
            csv_buffer = io.StringIO()
            csv_buffer.write("id,start,end,label,score,stage\n")
            for ev in refined_events:
                csv_buffer.write(
                    f"{uuid.uuid4().hex},{ev.start:.3f},{ev.end:.3f},{ev.label},{ev.score:.3f},{ev.stage}\n"
                )
            st.download_button(
                "下載偵測結果 CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="sed_events.csv",
                mime="text/csv",
            )

        st.subheader("3) 視覺化與互動")
        if show_spectrogram:
            pie_fig = build_event_pie(refined_events)
            if isinstance(pie_fig, (alt.Chart, alt.LayerChart, alt.ConcatChart, alt.HConcatChart, alt.VConcatChart, alt.FacetChart, alt.RepeatChart, alt.TopLevelMixin)):
                st.altair_chart(pie_fig, use_container_width=True)
            else:
                st.pyplot(pie_fig, clear_figure=True, use_container_width=True)

        st.markdown("**事件時間軸 (Stage1 / Stage2)**")
        df_events = events_to_df(stage1_events + refined_events)
        if not df_events.empty:
            stages = df_events["stage"].unique().tolist()
            labels = df_events["label"].unique().tolist()
            stage_filter = st.multiselect("篩選階段", options=stages, default=stages)
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
                st.info("無符合篩選條件的事件。")

        st.subheader("4) 如何換成真實模型？")
        st.markdown(
            """
        - 以 TorchScript 或 ONNX 載入你的 Stage-1 CNN，將 `Stage1CNNEdgeDetector.predict` 改為模型推論。
        - 將 Stage-2 換成已訓練的 Transformer/CRNN，輸入序列特徵或 logits，輸出事件列表。
        - 若需 Redis 緩衝，從 Stage-1 產生的 logits/特徵推入緩衝，再由 Stage-2 批次讀取。
        - 將告警管道（Webhook/SMS/Email）接在 Stage-2 結果上，依據閾值與冷卻時間推送。
        """
        )
    elif nav == "Applications":
        st.subheader("城市安全與應用場景 · 3D Map 互動")
        st.markdown(
            """
            在地圖上查看事件類型與強度，理解 GUARD 如何在城市中布署。
            - 事件越亮/柱狀越高代表置信度越高。
            - 可篩選類別、調整柱高比例與半徑。
            - 換成真實佈點：以實際 lat/lon 與 class/score 替換 `sample_geo_events()`.
            """
        )
        geo_df = sample_geo_events()
        classes = sorted(geo_df["class"].unique().tolist())
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sel_classes = st.multiselect("顯示類別", classes, default=classes)
        with col_ctrl2:
            radius = st.slider("半徑 (公尺)", 80, 300, 160, 10)
        with col_ctrl3:
            height_scale = st.slider("高度比例", 50, 300, 120, 10)

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
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="mapbox://styles/mapbox/dark-v11"))

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**交通與城市運維**")
            st.write(
                "- 十字路口警笛辨識，信號優先切換\n"
                "- 公車/地鐵站月台，尖叫/求救觸發安保\n"
                "- 工地/施工區域，異常爆裂聲即時通報"
            )
        with c2:
            st.markdown("**場域防護與民生**")
            st.write(
                "- 校園/醫院/商場的玻璃破裂與闖入偵測\n"
                "- 社區夜間巡防：異常撞擊聲、爭吵尖叫\n"
                "- 智慧建築：機房異音、設備異常噪音預警"
            )
    elif nav == "Case Studies":
        st.subheader("案例示意")
        st.markdown(
            """
            - 智慧街區：多點麥克風陣列，Stage-1 高召回，Stage-2 降誤報，誤報率 < 2%。
            - 校園防護：玻璃破裂與尖叫事件特化模型，事件到警報 < 2 秒。
            - 交通樞紐：警笛/撞擊辨識，與 CCTV 事件串接，提供事件回放。
            """
        )
    elif nav == "Support":
        st.subheader("支援與部署")
        st.markdown(
            """
            - 部署：`pip install -r requirements.txt` → `streamlit run app.py`（建議 Python 3.11）。
            - 模型：使用 `requirements-train.txt` 於本地訓練，導出 TorchScript/ONNX 後替換 app。
            - 系統需求：ffmpeg、可選 PyAudio/Redis；HTTPS/TLS 建議。
            - 疑難排除：缺少 torch 請切 3.10–3.12；ffmpeg 不在 PATH 請安裝並重開終端。
            """
        )
    else:  # Architecture
        st.subheader("Architecture · GUARD 端到端流程")
        st.markdown(
            """
            - PyAudio/ffmpeg：接收城市音訊。
            - Librosa：特徵抽取、切片（Log-Mel/PCEN）。
            - Stage-1 CNN：片段級高召回偵測。
            - Redis：特徵/logits 緩衝供序列模型取用。
            - Transformer/序列模型：時序精煉降誤報。
            - Deployment/Inference Service：輸出事件、警報與 API。
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
  <text x="60" y="205" class="title">🎙 PyAudio</text>
  <text x="60" y="230" class="desc">ffmpeg capture</text>

  <!-- Librosa -->
  <rect x="260" y="60" width="200" height="100" class="card"/>
  <rect x="280" y="85" width="46" height="20" rx="10" class="badgeBlue"/>
  <text x="280" y="125" class="title">🎵 Librosa</text>
  <text x="280" y="150" class="desc">Feature extract / slice</text>

  <!-- Stage1 CNN -->
  <rect x="260" y="220" width="200" height="100" class="card"/>
  <rect x="280" y="235" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="280" y="275" class="title">🔥 Stage-1 CNN</text>
  <text x="280" y="300" class="desc">Segment edge detect</text>

  <!-- Redis -->
  <rect x="520" y="135" width="200" height="120" class="card"/>
  <rect x="540" y="160" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="540" y="200" class="title">🧠 Redis Buffer</text>
  <text x="540" y="225" class="desc">Feature/logits cache</text>

  <!-- Transformer -->
  <rect x="780" y="60" width="200" height="100" class="card"/>
  <rect x="800" y="85" width="46" height="20" rx="10" class="badgeBlue"/>
  <text x="800" y="125" class="title">🌀 Transformer</text>
  <text x="800" y="150" class="desc">Sequence refine</text>

  <!-- Inference -->
  <rect x="780" y="220" width="200" height="100" class="card"/>
  <rect x="800" y="235" width="46" height="20" rx="10" class="badgeOrange"/>
  <text x="800" y="275" class="title">🚀 Inference Service</text>
  <text x="800" y="300" class="desc">Alerts / API</text>
  </g>
</svg>
"""
        st.markdown(svg_arch, unsafe_allow_html=True)
        st.caption("GUARD Pipeline — 逐節點顯示來源、前處理、兩階段模型、緩衝與服務")


if __name__ == "__main__":
    main()
