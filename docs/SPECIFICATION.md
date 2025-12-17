# GUARD Specification-Driven Development (SDD)

General Urban Audio Recognition & Defense — “The City Never Sleeps, Neither Do We.”

This document captures the specification to drive design, implementation, and validation. Use it to align scope, acceptance criteria, and testing before coding changes.

## 1. Vision & Scope
- **Goal:** Real-time, reliable detection of urban public-safety audio (e.g., gunshots, glass breaks, screams) with low false alarms and actionable alerts.
- **Scope (current):** Two-stage SED pipeline (Stage-1 CNN edge, Stage-2 sequence refiner), Streamlit demo UI, sample audio, Stage-1 training scaffold, observability stubs.
- **Out-of-scope (current):** Production-grade alert routing (SMS/email), SELD/DOA, multi-mic fusion, large-scale data labeling.

## 2. Stakeholders & Personas
- **Ops/Security Center**: Wants trustworthy alerts, low false alarms, dashboards.
- **City IT/SRE**: Needs deployability, observability, and runbooks.
- **Data/ML Engineer**: Needs training pipeline, reproducibility, clear interfaces.
- **Public/Government**: Trust via safety, privacy, and reliability commitments.

## 3. User Journeys / Use Cases
1) **Live monitoring**: Upload or stream audio, see detections and confidence, download CSV for audit.
2) **Model iteration**: Train Stage-1 CNN on NPZ log-mel data, export weights, plug into app.
3) **Scenario testing**: Use provided sample audios to validate thresholds and pipeline behavior.
4) **Ops readiness**: Inspect latency/health (placeholder), verify alert policies (cooldown/de-dup).

## 4. System Boundaries
- **Inputs:** Audio stream/file (mono), config (sample rate, frame/hop, thresholds), optional Redis for buffering (not mandatory in demo).
- **Processing:** Librosa features → Stage-1 edge detector → (optional Redis) → Stage-2 refiner → alert/event aggregation.
- **Outputs:** Event list (label, score, time bounds), optional CSV download, visualizations, logs/metrics (to be integrated).

## 5. Functional Specification (current demo)
- **F1. Audio ingestion:** Upload WAV/OGG/FLAC/MP3 or use built-in synthetic audio.
- **F2. Preprocessing:** Log-Mel/PCEN-ready hooks; demo uses librosa log-mel for viz and heuristics.
- **F3. Stage-1 detection:** Lightweight CNN placeholder (heuristic in app; trainable CNN scaffold in `src/models/baseline_cnn.py`).
- **F4. Buffering:** Redis placeholder (not active in app); interface expected to cache features/logits for Stage-2.
- **F5. Stage-2 refinement:** Sequence smoother placeholder (heuristic) to re-label and extend events.
- **F6. UI/UX:** Streamlit app with controls (sample rate, frame/hop, thresholds, class map), tables for Stage-1/2 events, spectrogram toggle, CSV download.
- **F7. Samples:** Generated test audio (`samples/demo_gunshot_glass.wav`, `samples/demo_noise.wav`) via `scripts/generate_sample_audio.py`.
- **F8. Training:** Stage-1 training script over NPZ log-mel dataset (`src/train/train_cnn.py`) with dataset wrapper (`src/train/dataset.py`).
- **F9. Docs:** README, STREAMLIT, TRAINING, PROMPT, SPEC.

## 6. Non-Functional Requirements (targets)
- **NFR-1 Latency:** Stage-1 decision < 250 ms per segment on CPU (demo heuristic meets); Stage-2 windowing within user-defined hop.
- **NFR-2 Reliability:** Graceful handling of missing audio/invalid files; no crashes on user input.
- **NFR-3 Observability (roadmap):** Expose counters for events, latency, Redis depth (when enabled); structured logs.
- **NFR-4 Security/Privacy:** No persistent storage of full audio by default; transport via HTTPS when deployed; restrict alert endpoints.
- **NFR-5 Configurability:** Thresholds, window sizes, class maps editable without code changes.
- **NFR-6 Portability:** Runs on Python 3.10+; minimal OS-specific assumptions (ffmpeg, optional PyAudio).

## 7. Acceptance Criteria (demo)
- AC1: User can start Streamlit via `streamlit run app.py` and load either uploaded audio or the built-in synthetic sample.
- AC2: Stage-1 and Stage-2 tables render with start/end/label/score/stage; CSV download works when events exist.
- AC3: Spectrogram renders when enabled; toggling off hides it without errors.
- AC4: Side-panel controls update detections on rerun (frame/hop, thresholds, class map).
- AC5: Sample generator script produces two WAV files in `samples/`.
- AC6: Training script runs on NPZ data and saves a checkpoint without code edits (given valid dataset).
- AC7: README links to all major flows and states the GUARD slogan.

## 8. Interfaces / Contracts (lightweight)
- **NPZ dataset:** keys `feat` (1 x MELS x T float32), `label` (int).
- **Detection event (app):** `{start: float, end: float, label: str, score: float, stage: str}`.
- **App inputs:** audio bytes; configs from UI; optional future Redis URL (not yet wired).
- **Model swap points:** `Stage1CNNEdgeDetector.predict` → replace with TorchScript/ONNX; `Stage2SequenceRefiner.refine` → replace with Transformer/CRNN outputs.

## 9. Data, Privacy, Safety
- Keep only derived features for model use; do not store raw audio beyond session in demo.
- Enforce retention limits and access control in production (not implemented in demo).
- For public deployment, add consent/notice for audio capture; secure alert channels with auth/signature.

## 10. Observability & Ops (roadmap hooks)
- Metrics to add: event_count by class/stage, latency_ms by stage, queue_depth (Redis), drop_ratio.
- Logs: structured JSON with model version, audio ref, event summary.
- Health: ffmpeg/PyAudio presence, Redis connectivity, model load success.

## 11. Testing Strategy
- **Unit:** Model forward (baseline CNN), dataset loading, heuristic detectors.
- **Integration:** End-to-end app run with sample audio; verify events and CSV content.
- **Artifacts:** `samples/*.wav` for smoke tests; NPZ fixtures for training.
- **Future:** Latency regression, false-alarm stress with noise corpus, Redis buffering tests.

## 12. Deployment Guidance (demo → prod)
- Demo: `streamlit run app.py`; headless with `--server.headless true --server.enableCORS false`.
- Prod: behind reverse proxy with TLS; add auth to UI if exposed; attach metrics exporter.
- Model export: TorchScript/ONNX for Stage-1/Stage-2 to reduce dependencies.

## 13. Traceability Checklist
- Branding & trust: README/app show GUARD name/slogan ✅
- Demo UX: upload + synthetic + controls + tables + spectrogram ✅
- Samples: generator + stored WAVs ✅
- Training scaffold: CNN + dataset + script + docs ✅
- Spec & guides: SPEC, STREAMLIT, TRAINING, PROMPT ✅
- Gaps to implement: real Stage-2 model, Redis pipeline, observability endpoints, alert channels.
