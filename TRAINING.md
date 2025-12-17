# Stage-1 CNN Training Scaffold

此檔說明如何使用本專案的 Stage-1 CNN 示意訓練腳本與資料格式。

## 前置
- 安裝依賴：`pip install -r requirements.txt`
- 準備 log-mel 特徵資料集（預先切片好的片段），每筆資料存成 `.npz`：
  - `feat`: shape `(1, n_mels, time)`，float32
  - `label`: 整數類別 id

## 產生示例資料（僅示意）
若暫無資料，可自行以 librosa 載入 WAV，產生 log-mel 後存成 NPZ。示例程式（請按需修改）：
```python
import pathlib, numpy as np, librosa
from pathlib import Path

def to_npz(wav, label, out_path, sr=16000, n_mels=64):
    y, _ = librosa.load(wav, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    np.savez(out_path, feat=logmel[None, ...], label=label)

out_dir = Path("data_npz"); out_dir.mkdir(exist_ok=True)
to_npz("samples/demo_gunshot_glass.wav", label=0, out_path=out_dir/"gunshot_glass.npz")
to_npz("samples/demo_noise.wav", label=1, out_path=out_dir/"noise.npz")
```

## 進行訓練
```bash
python -m src.train.train_cnn --data-dir data_npz --num-classes 2 --epochs 5 --batch-size 16
```
輸出模型會儲存到 `checkpoints/stage1_cnn.pt`（可自訂）。將其轉換成 TorchScript/ONNX 後，嵌入 `app.py` 的 Stage-1 推論。

## 模型說明
- `src/models/baseline_cnn.py`：輕量卷積網路，輸入 `(B, 1, MELS, T)`，輸出 logits。
- `src/train/train_cnn.py`：最小化訓練腳本，使用 CrossEntropyLoss。可加 class weighting 或 Focal Loss。
- `src/train/dataset.py`：讀取 `.npz` 的 dataset 封裝。

## 下一步
- 加入資料增強（SpecAugment、噪音混合、時間拉伸）。
- 導出 TorchScript/ONNX：使用 `torch.jit.trace` 或 `torch.onnx.export`。
- Stage-2：在 Transformer/CRNN 端做序列精煉，讀取 Stage-1 logits 序列。
