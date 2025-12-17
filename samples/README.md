# Samples

使用 `scripts/generate_sample_audio.py` 產生示例音檔：
- `demo_gunshot_glass.wav`：含槍響與玻璃破裂的合成音（約 8s）。
- `demo_noise.wav`：環境噪音樣本（約 10s）。
- `sample_00`~`sample_49`：批次合成的短片段（約 4s），涵蓋槍響/玻璃、警笛、環境噪音等混合。

執行：
```bash
python scripts/generate_sample_audio.py --count 50 --duration 4
```

產生的檔案可直接在 `app.py` 的 Streamlit 介面上傳測試。若需真實資料，請替換為實際錄音並遵循隱私與授權要求。
