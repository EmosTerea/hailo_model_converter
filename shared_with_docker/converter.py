#!/usr/bin/env python3
"""
ONNX → HEF converter for Hailo8 using Hailo SDK (DFC flow).

This script follows the Hailo tutorials (DFC_1/2/3 notebooks):
  1) Parse ONNX to a Hailo model HAR
  2) Optimize + quantize using a calibration set (NPY/NPZ)
  3) Compile to HEF for Hailo8

Defaults assume the current folder contains:
  - best.onnx
  - an images folder for calibration (e.g. ./images)

You can override paths via CLI flags. Examples:
  python converter.py \
    --onnx best.onnx \
    --model-name best \
    --calib-folder ./images \
    --img-size 640 640 \
    --preproc letterbox \
    --normalize-mean 0 --normalize-std 255 \
    --hw-arch hailo8

Outputs (by default, in the current directory):
  - <model-name>_hailo_model.har
  - <model-name>_quantized_model.har
  - <model-name>_compiled_model.har
  - <model-name>.hef
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

try:
    from hailo_sdk_client import ClientRunner
except Exception as e:  # pragma: no cover
    print("[error] Failed to import hailo_sdk_client.ClientRunner. Is Hailo SDK installed and activated?", file=sys.stderr)
    raise


## Removed .npy/.npz calibration support by request.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX to HEF via Hailo SDK (parse → optimize → compile)")
    parser.add_argument("--onnx", type=Path, default=Path("./best.onnx"), help="Path to ONNX model file")
    parser.add_argument("--model-name", type=str, default="best", help="Logical model name to use in artifacts")
    parser.add_argument("--calib-folder", type=Path, default=None, help="Folder with images to build calibration set on-the-fly")
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"), default=None, help="Target input size (height width)")
    parser.add_argument("--preproc", type=str, choices=["resize", "center-crop", "shorter-center-crop", "letterbox"], default="resize", help="Image preprocessing mode for folder → calib")
    parser.add_argument("--grayscale", action="store_true", help="Treat images as single-channel")
    parser.add_argument("--file-exts", type=str, default=".jpg,.jpeg,.png,.bmp", help="Comma-separated image extensions to include")
    # Removed saving .npy calibration blobs; always read from images
    parser.add_argument("--normalize-mean", type=str, default=None, help="Comma-separated means per channel to subtract (e.g. 123.675,116.28,103.53)")
    parser.add_argument("--normalize-std", type=str, default=None, help="Comma-separated stds per channel to divide (e.g. 58.395,57.12,57.375)")
    parser.add_argument("--letterbox-pad", type=int, default=114, help="Pad value for letterbox preproc (0-255)")
    parser.add_argument("--hw-arch", type=str, default="hailo8", help="Target HW arch (e.g. hailo8)")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for HAR/HEF")
    # Model script injection (optional)
    parser.add_argument(
        "--insert-norm-layer",
        action="store_true",
        help=(
            "Insert a model-script normalization layer so scaling happens on the neural core. "
            "When set, host-side calibration normalization is skipped. By default uses means=[0,0,0] and stds=[255,255,255] "
            "(or [0],[255] for grayscale). If --normalize-mean/--normalize-std are provided, they are used for the layer instead."
        ),
    )
    parser.add_argument(
        "--compiler-opt-max",
        action="store_true",
        help=(
            "Set compiler optimization level to 'max' via model script: "
            "performance_param(compiler_optimization_level=max). May increase compilation time."
        ),
    )
    # Optional ONNX parsing hints; usually not required.
    parser.add_argument("--start-nodes", nargs="*", default=None, help="Optional ONNX start node names")
    parser.add_argument("--end-nodes", nargs="*", default=None, help="Optional ONNX end node names")
    parser.add_argument("--net-input-shape", type=str, default=None,
                        help="Optional net input shape mapping, e.g. input.1:1,3,224,224")
    parser.add_argument("--max-calib", type=int, default=256, help="Cap calibration entries (0 = use all)")
    return parser.parse_args()


def _parse_net_input_shape(arg: Optional[str]) -> Optional[Dict[str, list]]:
    if not arg:
        return None
    # Format: "name:1,3,224,224[;name2:...]"
    mapping: Dict[str, list] = {}
    for part in arg.split(";"):
        if not part:
            continue
        name, shape_str = part.split(":", 1)
        shape = [int(x) for x in shape_str.split(",")]
        mapping[name] = shape
    return mapping


def main() -> int:
    args = parse_args()
    outdir: Path = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    onnx_path: Path = args.onnx.resolve()
    if not onnx_path.exists():
        print(f"[error] ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2

    calib_folder: Optional[Path] = args.calib_folder
    if calib_folder is None:
        print("[error] --calib-folder is required (point to a directory with calibration images).", file=sys.stderr)
        return 2

    model_name: str = args.model_name
    hw_arch: str = args.hw_arch

    hailo_model_har = outdir / f"{model_name}_hailo_model.har"
    quantized_model_har = outdir / f"{model_name}_quantized_model.har"
    compiled_model_har = outdir / f"{model_name}_compiled_model.har"
    hef_path = outdir / f"{model_name}.hef"

    # 1) Parse ONNX → HAR
    print(f"[info] Parsing ONNX → HAR | model={onnx_path.name} hw_arch={hw_arch}")
    runner = ClientRunner(hw_arch=hw_arch)

    net_input_shapes = _parse_net_input_shape(args.net_input_shape)
    translate_kwargs: Dict[str, Any] = {"onnx_path": str(onnx_path), "model_name": model_name}
    # Build call like the tutorials: runner.translate_onnx_model(path, name, ...)
    # but pass only optional hints if provided.
    if args.start_nodes:
        translate_kwargs["start_node_names"] = args.start_nodes
    if args.end_nodes:
        translate_kwargs["end_node_names"] = args.end_nodes
    if net_input_shapes:
        translate_kwargs["net_input_shapes"] = net_input_shapes

    # Signature in notebooks: runner.translate_onnx_model(onnx_path, onnx_model_name, ...)
    try:
        hn, npz = runner.translate_onnx_model(
            translate_kwargs.pop("onnx_path"),  # type: ignore[arg-type]
            translate_kwargs.pop("model_name"),  # type: ignore[arg-type]
            **translate_kwargs,
        )
    except Exception as e:
        # If the SDK recommends specific end nodes, auto‑retry with them.
        msg = str(e)
        marker = "Please try to parse the model again, using these end node names:"
        if marker in msg:
            # Extract the suggested list following the marker (may be comma/space separated)
            suggested = msg.split(marker, 1)[1].strip()
            # Take only the first line to avoid trailing logs
            suggested_line = suggested.splitlines()[0].strip()
            # Split by comma/space and keep non‑empty tokens
            end_nodes_retry = [tok.strip() for tok in suggested_line.replace(" ", "").split(",") if tok.strip()]
            if end_nodes_retry:
                print(f"[warn] Parse failed; retrying with recommended end nodes: {end_nodes_retry}")
                # Rebuild kwargs because we popped entries
                translate_kwargs_retry: Dict[str, Any] = {"end_node_names": end_nodes_retry}
                if args.start_nodes:
                    translate_kwargs_retry["start_node_names"] = args.start_nodes
                if net_input_shapes:
                    translate_kwargs_retry["net_input_shapes"] = net_input_shapes
                hn, npz = runner.translate_onnx_model(str(onnx_path), model_name, **translate_kwargs_retry)
            else:
                raise
        else:
            raise
    # Optional: inject normalization layer via model script so runtime normalization runs on the neural core
    if args.insert_norm_layer:
        # Choose channel count: prefer explicit grayscale flag, else assume 3
        num_channels = 1 if args.grayscale else 3
        # Resolve means/stds for the layer: use user-provided normalize args if present, else defaults (YOLO: mean=0, std=255)
        if args.normalize_mean:
            means = [float(x) for x in args.normalize_mean.split(',')]
        else:
            means = [0.0] * num_channels
        if args.normalize_std:
            stds = [float(x) for x in args.normalize_std.split(',')]
        else:
            stds = [255.0] * num_channels
        # Sanity: broadcast single value to channels if needed
        if len(means) == 1:
            means = means * num_channels
        if len(stds) == 1:
            stds = stds * num_channels
        if len(means) != num_channels or len(stds) != num_channels:
            print("[error] --insert-norm-layer requires means/stds to match channel count (1 or 3)", file=sys.stderr)
            return 2
        model_script = (
            f"normalization1 = normalization([{', '.join(str(m) for m in means)}], "
            f"[{', '.join(str(s) for s in stds)}])\n"
        )
        print(f"[info] Inserting normalization layer via model script: means={means}, stds={stds}")
        runner.load_model_script(model_script)

    runner.save_har(str(hailo_model_har))
    print(f"[info] Saved HAR (parsed): {hailo_model_har}")

    # 2) Optimize + quantize
    # Build or load calibration set
    # Always use images from --calib-folder; scan recursively
    calib_folder = calib_folder.resolve()
    if args.img_size is None:
        print("[error] --img-size H W is required when using --calib-folder.", file=sys.stderr)
        return 2

    H, W = args.img_size
    exts = tuple(x.strip().lower() for x in args.file_exts.split(",") if x.strip())
    paths = [p for p in sorted(calib_folder.rglob('*')) if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        print(f"[error] No images with extensions {exts} found under {calib_folder} (recursive)", file=sys.stderr)
        return 2
    if args.max_calib and args.max_calib > 0:
        paths = paths[: args.max_calib]

        from PIL import Image

        def _letterbox(img: np.ndarray, out_h: int, out_w: int, pad: int) -> np.ndarray:
            h, w = img.shape[:2]
            r = min(out_w / w, out_h / h)
            new_w, new_h = int(round(w * r)), int(round(h * r))
            img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR))
            canvas = np.full((out_h, out_w, img.shape[2]), pad, dtype=img_resized.dtype)
            top = (out_h - new_h) // 2
            left = (out_w - new_w) // 2
            canvas[top:top + new_h, left:left + new_w] = img_resized
            return canvas

        def _center_crop(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
            h, w = img.shape[:2]
            top = max(0, (h - out_h) // 2)
            left = max(0, (w - out_w) // 2)
            return img[top: top + out_h, left: left + out_w]

        def _shorter_side_then_center_crop(img: np.ndarray, out_h: int, out_w: int, resize_side: int = 256) -> np.ndarray:
            h, w = img.shape[:2]
            scale = resize_side / min(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR))
            return _center_crop(img_resized, out_h, out_w)

        def _resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
            return np.array(Image.fromarray(img).resize((out_w, out_h), Image.Resampling.BILINEAR))

    c = 1 if args.grayscale else 3
    calib_dataset = np.empty((len(paths), H, W, c), dtype=np.float32)

    for i, p in enumerate(paths):
            img = Image.open(p)
            if args.grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            arr = np.array(img)
            if args.grayscale and arr.ndim == 2:
                arr = arr[:, :, None]

            if args.preproc == "letterbox":
                arr = _letterbox(arr, H, W, pad=int(args.letterbox_pad))
            elif args.preproc == "center-crop":
                # First resize so that the shorter side >= target, then center-crop
                h, w = arr.shape[:2]
                scale = max(H / h, W / w)
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                arr = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.BILINEAR))
                arr = _center_crop(arr, H, W)
            elif args.preproc == "shorter-center-crop":
                arr = _shorter_side_then_center_crop(arr, H, W)
            else:  # resize
                arr = _resize(arr, H, W)

            arr = arr.astype(np.float32)

            # Host normalization: if --insert-norm-layer is OFF, apply either user-provided
            # means/stds or YOLO defaults (mean=0, std=255). If --insert-norm-layer is ON, skip here.
            if not args.insert_norm_layer:
                if args.normalize_mean:
                    means = [float(x) for x in args.normalize_mean.split(',')]
                    stds = [float(x) for x in (args.normalize_std or '').split(',')] if args.normalize_std else None
                else:
                    means = [0.0]
                    stds = [255.0]
                if len(means) not in (1, c):
                    print("[error] --normalize-mean must have 1 or C values", file=sys.stderr)
                    return 2
                if stds is not None and len(stds) not in (1, c):
                    print("[error] --normalize-std must have 1 or C values", file=sys.stderr)
                    return 2
                if len(means) == 1:
                    arr = arr - means[0]
                else:
                    arr = arr - np.array(means, dtype=np.float32)
                if stds is not None:
                    if len(stds) == 1:
                        arr = arr / stds[0]
                    else:
                        arr = arr / np.array(stds, dtype=np.float32)

            calib_dataset[i] = arr

    # (Removed .npy/.npz handling)

    # Optionally cap number of calibration entries to reduce memory/time.
    def _cap_entries(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            arr = arr[None, ...]
        if args.max_calib and args.max_calib > 0 and arr.shape[0] > args.max_calib:
            return arr[: args.max_calib]
        return arr

    calib_dataset = _cap_entries(calib_dataset)

    print("[info] Optimizing + quantizing (this may take a few minutes)...")
    runner.optimize(calib_dataset)
    runner.save_har(str(quantized_model_har))
    print(f"[info] Saved HAR (quantized): {quantized_model_har}")

    # 3) Compile → HEF
    print("[info] Compiling to HEF for Hailo device...")
    runner = ClientRunner(har=str(quantized_model_har))
    # Optional: set compiler optimization level to max
    if args.compiler_opt_max:
        runner.load_model_script("performance_param(compiler_optimization_level=max)\n")
        print("[info] Compiler optimization level set to: max")

    hef = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"[info] Saved HEF: {hef_path}")

    runner.save_har(str(compiled_model_har))
    print(f"[info] Saved HAR (compiled): {compiled_model_har}")
    print("[done] Conversion completed successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
