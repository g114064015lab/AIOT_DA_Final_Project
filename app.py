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

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st


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


# --------- Streamlit UI -------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="GUARD | Urban Acoustic SED & Alert Demo",
        page_icon="🎧",
        layout="wide",
    )
    st.title("GUARD：城市聲音事件偵測與公共安全警報 — 互動 Demo")
    st.markdown("**Slogan：GUARD: The City Never Sleeps, Neither Do We.**")
    st.caption("General Urban Audio Recognition & Defense — 守護與防禦，強調系統安全性與可靠性。")
    st.caption("Two-Stage SED (CNN → Transformer/CRNN) with Librosa preprocessing. "
               "Upload或使用合成範例音訊，調整閾值與時序設定，查看偵測結果。")

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

    st.subheader("1) 載入音訊")
    uploaded = st.file_uploader("上傳 WAV/OGG/FLAC/MP3", type=["wav", "ogg", "flac", "mp3"])
    use_demo = st.checkbox("使用內建合成範例音訊（含槍響+玻璃破裂）", value=uploaded is None)
    audio_bytes: bytes | None = None
    audio_np: np.ndarray | None = None

    if uploaded is not None:
        audio_bytes = uploaded.read()
        audio_np, sr = load_audio(io.BytesIO(audio_bytes), sample_rate=sr)
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

    st.subheader("3) 音訊視覺化")
    if show_spectrogram:
        fig = plot_spectrogram(audio_np, sr)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    st.subheader("4) 如何換成真實模型？")
    st.markdown(
        """
- 以 TorchScript 或 ONNX 載入你的 Stage-1 CNN，將 `Stage1CNNEdgeDetector.predict` 改為模型推論。
- 將 Stage-2 換成已訓練的 Transformer/CRNN，輸入序列特徵或 logits，輸出事件列表。
- 若需 Redis 緩衝，從 Stage-1 產生的 logits/特徵推入緩衝，再由 Stage-2 批次讀取。
- 將告警管道（Webhook/SMS/Email）接在 Stage-2 結果上，依據閾值與冷卻時間推送。
"""
    )


if __name__ == "__main__":
    main()
