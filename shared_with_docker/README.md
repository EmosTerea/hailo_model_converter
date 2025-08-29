# Hailo Model Converter

Convert ONNX models to Hailo8 HEF with a calibration‑set builder for YOLO datasets.

## What’s Inside
- **Calibration builder:** Picks a diverse, class‑balanced subset of images from a YOLO dataset (symlinks + `manifest.json`).
- **ONNX→HEF pipeline:** Parse ONNX → optimize/quantize with your images → compile to HEF using the Hailo SDK.

## Prerequisites
- **Hailo AI SW Suite/SDK:** Prefer Docker. Matching docs live in `doc/`.
- **Dataset:** YOLO format with `train/`, `val/` (and optional `test/`), each with `images/` and `labels/`.
- **Model:** An ONNX file and its expected input size.
- **VS Code (optional):** To attach to the container.

## Quick Start
1) Start the Hailo AI SW Suite container (e.g., via your `hailo_ai_sw_suite_docker_run.sh`).
2) Attach with VS Code → Dev Containers → Attach to Running Container… → open `/local/` (shared) or `/local/shared_with_docker`.
3) Copy artifacts into the shared folder:
   - `model.onnx`
   - `Archive/` (your YOLO dataset root)
4) Build a calibration set (example: 1024 images, prefer `val`):
   ```bash
   python build_calibration_set.py \
     --archive-root Archive --outdir calib_yolo \
     --n 1024 --prefer val --min-per-class 12 --max-per-seq 6
   ```
5) Convert ONNX → HEF (640×640 letterbox; cap 1024 calib images):
   ```bash
   python converter.py \
     --onnx model.onnx --model-name model \
     --calib-folder ./calib_yolo \
     --img-size 640 640 --preproc letterbox --letterbox-pad 114 \
     --net-input-shape images:1,3,640,640 \
     --max-calib 1024
   ```
   If parsing needs explicit end nodes, add:
   ```bash
   --end-nodes \
   "/model.22/cv2.0/cv2.0.2/Conv" \
   "/model.22/cv3.0/cv3.0.2/Conv" \
   "/model.22/cv2.1/cv2.1.2/Conv" \
   "/model.22/cv3.1/cv3.1.2/Conv" \
   "/model.22/cv2.2/cv2.2.2/Conv"
   ```

## Outputs
- `<model>_hailo_model.har` (parsed)
- `<model>_quantized_model.har` (optimized/quantized)
- `<model>_compiled_model.har` (post‑compile)
- `<model>.hef` (deploy to Hailo8)

## Script Highlights
- **build_calibration_set.py:**
  - Ensures per‑class image coverage (`--min-per-class`) and caps near‑duplicates by sequence (`--max-per-seq`).
  - Prefers one split (`--prefer val|train`); can backfill from others (`--include-test`).
  - Emits symlinks (copy fallback) plus `manifest.json` in `--outdir`.
- **converter.py:**
  - Calibrates directly from images in `--calib-folder` (recursive; filter via `--file-exts`).
  - Preprocessing: `resize`, `center-crop`, `shorter-center-crop`, `letterbox` (`--img-size H W`, `--letterbox-pad`).
  - Normalization: host side via `--normalize-mean/--normalize-std` (YOLO default 0/255), or push into the model with `--insert-norm-layer`.
  - Parsing hints: `--start-nodes`, `--end-nodes`, `--net-input-shape name:1,3,H,W`.
  - Compiler: `--compiler-opt-max` to set optimization level to "max".
  - Control size/time: `--max-calib` caps calibration images used.

## Expected Tree (after running)
```bash
shared_with_docker
├── Archive
│   ├── generated_dataset_01.yaml
│   ├── test/{images,labels}
│   ├── train/{images,labels}
│   └── val/{images,labels}
├── build_calibration_set.py
├── calib_yolo/              # selected images + manifest.json
├── converter.py
├── doc/
├── model_hailo_model.har
├── model_quantized_model.har
├── model_compiled_model.har
├── model.hef
└── model.onnx
```

## Troubleshooting
- Parse error suggesting end nodes: the script may auto‑retry; otherwise pass them with `--end-nodes` as shown above.
- No images found: check `--calib-folder` path and extensions via `--file-exts`.
- Shape mismatch: ensure `--img-size` matches the model and set `--net-input-shape` if the ONNX input is dynamic.
