# Maternin Anemia Triage — Hugging Face Space Demo — Design Spec

**Date:** 2026-08-04
**Status:** Implemented
**Repo:** `~/hf-spaces/maternin-anemia-triage/`
**Target Space:** `huggingface.co/<handle>/maternin-anemia-triage`

## Goal

Ship a public Gradio demo that showcases the Maternin AI service's
computer-vision anemia triage pipeline. Visitors upload or take a photo
of a conjunctiva (inner eyelid); the demo returns a Normal / Anemia-risk
label plus confidence, with the eye ROI drawn on the photo.

## Approach (decided up-front)

Single-model **MobileNetV3-Small** (`anemia_mobilenetv3_v2_real.onnx`,
F1 = 0.83, ~6 MB total with external weights) — chosen because:

- ConvNeXt-Tiny is ~112 MB total, doesn't fit the HF Spaces free-tier
  50 MB repo limit without Git LFS, and the inference improvement
  (~6 F1 points) doesn't justify the friction for a demo.
- MobileNetV3 inference on CPU is fast enough for an interactive demo
  (~50–150 ms per prediction on Spaces' CPU).

The README documents how to swap in ConvNeXt-Tiny for a higher-accuracy
build (paid tier or with LFS).

## Files

| Path | Purpose |
|------|---------|
| `app.py` | Gradio UI: upload + webcam inputs, ROI overlay output, label/confidence markdown |
| `infer.py` | Inference module: MediaPipe (optional) + ONNX Runtime + softmax. Mirrors production constants. |
| `anemia_mobilenetv3_v2_real.onnx` (+ `.data` + `_metadata.json`) | Trained model artifact |
| `samples/` | 3 generated PIL placeholder images (clearly labeled as placeholders so visitors know to use real photos) |
| `generate_samples.py` | Regenerates placeholders |
| `requirements.txt` | Pinned deps for Spaces |
| `README.md` | Frontmatter (sdk/title/license) + human docs |

## Inference contract

Mirrors `ai-service/app/training/anemia_real_train.py`:

- Input: PIL image ≥ 50×50 px
- ROI: MediaPipe Face Mesh palpebral landmarks + 30% padding, **falling
  back to a center-upper heuristic crop** when MediaPipe is unavailable
  or no face is detected. Fallback is essential because MediaPipe
  ≥0.10.10 removed the legacy `mp.solutions.face_mesh` API used by
  production code.
- Preprocess: RGB → resize 224×224 → ImageNet normalize
  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) → CHW float32
- Output: softmax over 2 logits → `[P(normal), P(anemia)]`
- Decision: argmax with **60% confidence gate**; below threshold →
  `"Uncertain — retake photo"`.

## UI shape

Single-page Gradio Blocks layout, two columns:

- **Left:** image input (upload + webcam + example gallery), Analyze
  button
- **Right:** annotated image (with green ROI rectangle), label markdown
  with confidence + per-class probabilities

Top: disclaimer that this is a triage aid, not a diagnosis.
Bottom: model/pipeline summary.

## Edge cases handled

- No face detected → fall back to center crop, label prediction as-is,
  show note in result
- MediaPipe missing / API mismatch → fall back to center crop with a
  different note
- Image too small → explicit "Image too small (minimum 50×50 pixels)"
  message instead of a fake prediction
- Confidence < 60% → "Uncertain — retake photo" instead of a confident
  wrong label
- Model file missing at startup → `app.py` exits with a clear FATAL
  message rather than a stack trace

## Out of scope

- Multi-model comparison tab (single-model demo by design)
- Login, database, callbacks
- Production routing/aggregation logic (preeclampsia, risk_aggregator)
- MediaPipe Tasks API migration (kept the legacy constant set so this
  mirrors production; the fallback covers the API-drift case)

## Latent issue surfaced

`ai-service/app/models/landmark_roi.py` uses the same
`mp.solutions.face_mesh` API. With the current venv on MediaPipe
0.10.35, this code path would fail if called. **Not yet fixed in
production** — production's inference entry point (`inference.py`) is
lazy-imported and isn't currently hit. Worth a separate ticket.
