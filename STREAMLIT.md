# Streamlit Demo 指南

此檔說明如何啟動並體驗 `app.py` 的互動式聲音事件偵測 Demo，對應架構圖 `Arch.png`，展示 Librosa 前處理、Stage-1 CNN 邊緣偵測（示意）、Stage-2 Transformer/CRNN 精煉（示意）與 UI 呈現。

## 快速開始
1) 建立虛擬環境並安裝依賴
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) 安裝 ffmpeg（確保 `ffmpeg` 在 PATH）
3) 執行 Demo
```bash
streamlit run app.py
```
4) 瀏覽器開啟自動跳轉的本地網址（預設 http://localhost:8501）

## 使用方式
- 上傳 WAV/OGG/FLAC/MP3，或勾選「使用內建合成範例音訊」。
- 在側邊欄調整：
  - 取樣率、frame/hop 長度
  - Stage-1 energy 閾值（邊緣偵測敏感度）
  - Stage-2 score bonus、最小事件長度
  - 類別映射（用於示意再標記）
- 主區塊會顯示：
  - 階段 1/2 偵測結果表格（可下載 CSV）
  - 選擇性頻譜圖
  - 如何替換為真實模型的指引

## 將示意換成真實模型
- Stage-1：在 `Stage1CNNEdgeDetector.predict` 中載入 TorchScript/ONNX CNN，輸入特徵/切片後輸出 logits，再映射事件區段。
- Stage-2：將 `Stage2SequenceRefiner` 替換成 Transformer/CRNN，讀取序列特徵或 Stage-1 logits，做時序平滑與最終分類。
- Redis：若需模擬實際流水線，將 Stage-1 輸出寫入 Redis，Stage-2 以滑動窗口批次讀取。
- 告警：在 Stage-2 結果後掛上 Webhook/SMS/Email，加入冷卻時間與去抖動。

## 部署建議
- 開發模式：`streamlit run app.py`，關閉 `--server.runOnSave=false` 以避免自動重啟時中斷推論。
- 生產模式：以 `streamlit run app.py --server.headless true --server.enableCORS false`，搭配反向代理（nginx）與 TLS。
- 監控：啟用 Streamlit 日誌輸出，或將模型延遲、事件量寫入 Prometheus（需額外程式碼）。

## 常見問題
- 沒有音訊輸入：請先上傳檔案或勾選合成範例。
- 缺少 ffmpeg：安裝後重新開啟終端，確認 `ffmpeg -version` 可用。
- 相依版本衝突：`pip install --upgrade pip setuptools wheel` 後重新安裝。
