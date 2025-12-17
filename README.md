# 城市聲音事件偵測與公共安全警報系統

兩階段聲音事件偵測（Two-Stage SED）架構，結合輕量 CNN 進行邊緣偵測與 Transformer/CRNN 進行時序精煉，目標是智慧城市場景下的即時公共安全警報（例如槍響）。本 README 說明系統組件、開發流程與部署重點，對應架構圖 `Arch.png`。

## 系統架構
- **音訊擷取**：PyAudio via ffmpeg，固定片段長度（例如 1–2s），重疊滑窗。
- **前處理**：Librosa 提取特徵（Log-Mel/PCEN、時頻切片、增益/歸一化）。
- **第一階段（CNN）**：PyTorch 輕量 CNN 做片段級分類，負責高召回、低延遲的「邊緣偵測」。
- **快取/序列緩衝**：Redis 暫存特徵與模型 logits，供後續時序模型讀取。
- **第二階段（Transformer/CRNN）**：對多片段序列進行時序精煉，提升定位與類別準確度。
- **服務層**：部署 API/推論服務，暴露事件流、健康檢查與警報通知接口。

## 核心設計原則
- **低延遲、可回放**：第一階段負責即時觸發，第二階段在可接受的時序窗口內精煉。
- **抗噪**：PCEN、頻段遮罩、SpecAugment；背景模型自適應校正。
- **可擴充**：事件類別可配置；支持在地化閾值與動態類別增量。
- **可觀測性**：事件計數、模型延遲、Redis 深度、丟包率、誤報/漏報率。

## 目錄與關鍵檔案（建議）
- `Arch.png`：高階架構。
- `configs/`：模型、特徵、推論/閾值與通知設定。
- `src/audio/`：擷取、切片、增益控制、靜音檢測。
- `src/features/`：Librosa 特徵（Log-Mel/PCEN）、標準化、增強。
- `src/models/`：CNN、Transformer/CRNN、匯入/匯出 ONNX 或 TorchScript。
- `src/pipeline/`：兩階段推論邏輯、Redis 交互。
- `src/service/`：API（REST/gRPC）、健康檢查、指標導出。
- `tests/`：單元/整合測試，含合成音與錄音片段樣例。

## 環境需求
- Python 3.10+；ffmpeg；PyAudio（需麥克風權限）。
- 依硬體：CPU 即時推論需優化；如可用 GPU/TPU 則啟用加速。
- 依服務：本地 Redis（可選）；雲端部署需設定網路與安全組。

## Streamlit 互動 Demo
- 依據 `requirements.txt` 安裝相依後執行：`streamlit run app.py`
- 側邊欄可調整取樣率、frame/hop、Stage-1 閾值、Stage-2 平滑參數；可上傳音檔或使用合成範例。
- Demo 目前內建示意模型（啟發式），請在 `app.py` 中替換為 TorchScript/ONNX CNN 與 Transformer/CRNN，即可接軌真實架構。
- 詳細說明請見 `STREAMLIT.md`。

### 安裝範例（本地開發）
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio \
    librosa soundfile pyaudio redis fastapi uvicorn[standard] \
    numpy scipy
```

安裝 ffmpeg（Windows 可用 choco 或直接安裝官方套件），確保 `ffmpeg` 指令可用。

## 執行流程（範例命令）
- **啟動 Redis（選用）**：`redis-server`
- **啟動推論服務**（假設 `src/service/app.py`）：\
  `uvicorn src.service.app:app --host 0.0.0.0 --port 8000`
- **本地麥克風即時推論**（假設 `src/pipeline/stream_infer.py`）：\
  `python -m src.pipeline.stream_infer --config configs/infer.yaml --mic-index 0`

## 設定要點
- `configs/infer.yaml`
  - `sample_rate`: 16000
  - `frame_len`: 2.0 (seconds), `hop_len`: 0.5
  - `features`: `logmel` 或 `pcen`, `n_mels`: 64/128
  - `stage1.thresholds`: 類別閾值（偏召回）
  - `stage2.window`: 序列長度（例如 5–10 片段）
  - `redis`: host/port/db，緩衝大小與 TTL
  - `alerts`: Webhook/Email/SMS；冷卻時間、重複抑制

## 模型訓練建議
- **資料**：城市環境錄音，包含槍響、破窗、車禍警示等；混入多樣噪音（車流、人群、雨、施工）。
- **增強**：SpecAugment、時間拉伸、隨機噪音混合、增益抖動、混響。
- **損失**：加權 BCE/Focal Loss（處理類別不均）；可加 CTC/匯流排損失處理時序對齊。
- **評估**：片段級 F1、事件級 F1、定位誤差、延遲分佈；分場景（室內/室外/車內）分別報告。
- **導出**：針對部署導出 TorchScript 或 ONNX，並記錄特徵正規化統計以便推論對齊。

## 推論與警報策略
- 第一階段低閾值觸發候選；第二階段基於序列重新評分，輸出最終事件。
- 支援多策略：穩定窗口投票、置信度平滑、冷卻時間、同類事件合併。
- 事件上報內容：類別、置信度、開始/結束時間、地點/裝置 ID、錄音片段引用。

## 監控與日誌
- 指標：`event_count{class=...}`、`latency_ms{stage=...}`、`redis_depth`、`drop_ratio`、`false_alarm_rate`。
- 日誌：使用結構化日誌（JSON），保留音訊片段引用與模型版本。
- 健康檢查：依賴 ffmpeg/PyAudio、Redis 可用性、模型載入、推論耗時。

## 安全與隱私
- 僅儲存必要片段；對錄音做存留時間限制與存取控管。
- 通訊加密（HTTPS/TLS）；敏感告警通道需驗證與簽名。
- 防止濫用：速率限制、IP 白名單、權限分級。

## 待辦與延伸
- 加入 SELD（同時預估方位）與麥克風陣列支援。
- 自動標註與半監督學習，提升低資源類別表現。
- 針對邊緣裝置的量化/剪枝與流水線優化。
- 事件重放與事後調查介面（配合監控系統）。
