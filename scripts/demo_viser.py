import pickle
import time
import argparse
import inspect
import math
import tempfile
import shutil
import yaml
import torch
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from natsort import natsorted
from typing import List, Optional, Sequence

from loger.utils.rotation import mat_to_quat
from loger.utils.geometry import depth_edge
from loger.models.pi3 import Pi3
from loger.utils.viser_utils import viser_wrapper


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg")


def is_video_file(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS


def _is_huggingface_model_name(model_name: str) -> bool:
    return model_name.startswith("yyfz233/")


def _maybe_parse_sequence(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
            except yaml.YAMLError:
                return value
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    return value


def _load_yaml_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {config_path}")

    return config


def _list_directory_images(input_path: Path, start_frame: int, end_frame: int, stride: int) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(str(path) for path in input_path.glob(pattern))

    filtered_paths = [
        image_path
        for image_path in natsorted(image_paths)
        if "depth" not in Path(image_path).name.lower()
    ]
    end_idx = end_frame if end_frame != -1 else None
    return filtered_paths[start_frame:end_idx:stride]


def _collect_input_frames(
    input_paths: Sequence[Path],
    start_frame: int,
    end_frame: int,
    stride: int,
) -> tuple[List[str], dict[str, int], dict[str, Path]]:
    all_image_names: List[str] = []
    input_indices: dict[str, int] = {}
    temp_frame_dirs: dict[str, Path] = {}

    for index, input_path in enumerate(input_paths):
        camera_key = f"cam{index:02d}"
        temp_key = f"input{index + 1}"
        if index > 0:
            input_indices[camera_key] = len(all_image_names)

        if is_video_file(input_path):
            temp_dir = Path(tempfile.mkdtemp(prefix=f"pi3_frames_{temp_key}_"))
            temp_frame_dirs[temp_key] = temp_dir
            current_frames = extract_frames_from_video(
                input_path,
                temp_dir,
                start_frame,
                end_frame,
                stride,
            )
        elif input_path.is_dir():
            current_frames = _list_directory_images(input_path, start_frame, end_frame, stride)
        else:
            raise ValueError(f"Input path must be a directory or supported video file: {input_path}")

        if not current_frames:
            temp_dir = temp_frame_dirs.pop(temp_key, None)
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir)
            input_indices.pop(camera_key, None)
            continue

        all_image_names.extend(current_frames)

    return all_image_names, input_indices, temp_frame_dirs


def _build_sequence_name(input_paths: Sequence[Path]) -> str:
    parts: List[str] = []
    for input_path in input_paths:
        if input_path.parent.name:
            parts.append(input_path.parent.name)
        parts.append(input_path.name)
    return "_".join(parts)


def _resolve_saved_predictions_path(load_path: Path, seq_name: Optional[str]) -> Path:
    if load_path.is_dir():
        if seq_name:
            seq_candidate = load_path / f"{seq_name}.pt"
            if seq_candidate.exists():
                return seq_candidate
        return load_path / "predictions.pt"
    return load_path


def _convert_tensor_tree_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _convert_tensor_tree_to_numpy(subvalue) for key, subvalue in value.items()}
    return value


def _build_forward_kwargs(args, config: dict) -> dict:
    training_settings = config.get("training_settings", {})
    model_settings = config.get("model", {})
    se3_from_config = model_settings.get("se3", config.get("se3", False))

    return {
        "window_size": args.window_size if args.window_size is not None else training_settings.get("window_size", -1),
        "overlap_size": args.overlap_size if args.overlap_size is not None else training_settings.get("overlap_size", 0),
        "reset_every": args.reset_every if args.reset_every is not None else training_settings.get("reset_every", 0),
        "num_iterations": config.get("num_iterations", 1),
        "sim3": config.get("sim3", False) or args.sim3,
        "sim3_scale_mode": args.sim3_scale_mode,
        "se3": args.se3 if args.se3 is not None else bool(se3_from_config),
        "turn_off_ttt": args.no_ttt,
        "turn_off_swa": args.no_swa,
    }


def _cleanup_temp_dirs(temp_frame_dirs: dict[str, Path]) -> None:
    for temp_dir_path in temp_frame_dirs.values():
        if temp_dir_path.exists():
            print(f"Cleaning up temporary directory: {temp_dir_path}")
            shutil.rmtree(temp_dir_path)


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    start_frame: int,
    end_frame: int,
    stride: int,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    saved_frame_count = 0
    image_paths = []

    actual_end_frame = total_frames - 1 if end_frame == -1 else end_frame
    if actual_end_frame >= total_frames:
        print(f"Warning: end_frame ({actual_end_frame}) is beyond total frames ({total_frames-1}). Adjusting to last frame.")
        actual_end_frame = total_frames - 1

    print(f"Extracting frames from {video_path}: start={start_frame}, end={actual_end_frame}, stride={stride}")

    while True:
        ret, frame = cap.read()
        if not ret or current_frame_idx > actual_end_frame:
            break

        if current_frame_idx >= start_frame and (current_frame_idx - start_frame) % stride == 0:
            frame_filename = f"frame_{saved_frame_count:06d}.png"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            image_paths.append(str(frame_path))
            saved_frame_count += 1
        
        current_frame_idx += 1

    cap.release()
    print(f"Successfully extracted {saved_frame_count} frames to {output_dir}.")
    return natsorted(image_paths)


def load_pi3_model(model_name: str, config: Optional[dict] = None, pi3x: bool = False, pi3x_metric: bool = True):
    """Initializes the Pi3 model and loads weights."""
    print("Initializing Pi3 model...")

    model_kwargs = {}
    if config:
        model_config = config.get("model", {})
        pi3_signature = inspect.signature(Pi3.__init__)
        valid_kwargs = {
            name
            for name, param in pi3_signature.parameters.items()
            if name not in {"self", "args", "kwargs"}
            and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        for key in sorted(valid_kwargs):
            if key in model_config:
                value = model_config[key]
                if key in {"ttt_insert_after", "attn_insert_after"}:
                    value = _maybe_parse_sequence(value)
                model_kwargs[key] = value

        print("Model parameters from config:", model_kwargs)

    if pi3x:
        model_kwargs["pi3x"] = True
        model_kwargs["pi3x_metric"] = pi3x_metric
        if model_name == "yyfz233/Pi3":
            print("Switching default model to yyfz233/Pi3X because --pi3x is set.")
            model_name = "yyfz233/Pi3X"

    # Initialize model with parameters from config
    model = Pi3(**model_kwargs)

    if _is_huggingface_model_name(model_name):
        print("Loading pre-trained weights from Hugging Face Hub...")
        model = model.from_pretrained(model_name, strict=False if pi3x else True, **model_kwargs)
        print("Model loaded successfully from Hugging Face Hub.")
        return model

    # Load pre-trained weights
    print(f"Loading pre-trained weights from: {model_name}")
    # Use strict=False to allow for architecture mismatches when loading weights
    # This is useful when the config defines a different architecture than the saved checkpoint
    checkpoint = torch.load(model_name, map_location="cpu", weights_only=False)
    # If the checkpoint is a state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Adjust state_dict keys if they are prefixed (e.g., by DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove `module.`
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    print("Model loaded successfully.")

    return model


def run_core_inference(
    model_obj: Pi3,
    input_paths: List[str],
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_resolution: Optional[List[int]] = None,
):
    """
    Handles data preparation and runs the core model inference for Pi3.
    """
    model_obj.eval()
    model_obj = model_obj.to(device)

    if target_resolution is None:
        target_resolution = [504, 280]

    path_objects = [Path(input_path) for input_path in input_paths]
    all_image_names, input_indices, temp_frame_dirs = _collect_input_frames(
        path_objects,
        start_frame,
        end_frame,
        stride,
    )

    if not all_image_names:
        print("Error: No images found from any input.")
        return None, [], {}, {}

    print(f"Loading images from combined inputs ({len(all_image_names)} images found)...")
    images_tensor = load_images_from_paths(
        all_image_names,
        Target_W=target_resolution[0],
        Target_H=target_resolution[1],
    ).to(device)
    print(f"Preprocessed images tensor shape: {images_tensor.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        raw_model_predictions = model_obj(images_tensor[None])

    # Post-process predictions
    raw_model_predictions["images"] = images_tensor[None].permute(0, 1, 3, 4, 2)
    raw_model_predictions["conf"] = torch.sigmoid(raw_model_predictions["conf"])
    edge = depth_edge(raw_model_predictions["local_points"][..., 2], rtol=0.03)
    raw_model_predictions["conf"][edge] = 0.0
    if "local_points" in raw_model_predictions:
        del raw_model_predictions["local_points"]

    return raw_model_predictions, all_image_names, input_indices, temp_frame_dirs


def _try_load_timestamps_for_images(image_paths, input_rgb_dir: Path):
    """Best-effort timestamp loader.

    Priority:
    1) <parent>/rgb.txt (TUM style: "timestamp rgb/xxxxx.png")
    2) <input_rgb_dir>/timestamps.txt (one timestamp per line)
    3) Fallback to sequential indices starting at 0
    """
    # 1) TUM-format rgb.txt in the parent directory
    if input_rgb_dir.is_file():
        return [float(i) for i in range(len(image_paths))]

    rgb_txt_path = input_rgb_dir.parent / "rgb.txt"
    if rgb_txt_path.exists():
        name_to_ts = {}
        with open(rgb_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                img_rel = parts[1]
                # Map by basename for robustness
                name_to_ts[Path(img_rel).name] = ts

        ts_list = []
        for p in image_paths:
            ts_list.append(name_to_ts.get(Path(p).name, None))
        if all(t is not None for t in ts_list) and len(ts_list) == len(image_paths):
            return ts_list
        # If partial or mismatch, fall through to next option

    # 2) timestamps.txt alongside images
    timestamps_txt = input_rgb_dir / "timestamps.txt"
    if timestamps_txt.exists():
        with open(timestamps_txt, "r") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
        # Take as many as needed in order
        ts_list = []
        for i in range(min(len(raw_lines), len(image_paths))):
            try:
                ts_list.append(float(raw_lines[i]))
            except ValueError:
                ts_list.append(float(i))
        # If fewer timestamps than images, pad with indices
        for i in range(len(ts_list), len(image_paths)):
            ts_list.append(float(i))
        return ts_list

    # 3) Fallback: sequential indices as timestamps
    return [float(i) for i in range(len(image_paths))]


def write_trajectory_txt(output_path: Path, timestamps, translations, quaternions):
    """Write trajectory file with lines: ts tx ty tz qx qy qz qw"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, t, q in zip(timestamps, translations, quaternions):
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def load_images_from_paths(image_paths, PIXEL_LIMIT=255000, Target_W=None, Target_H=None, verbose=True):
    sources = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as image:
                sources.append(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            print(f"Could not load image {img_path}: {exc}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if Target_W is None and Target_H is None:
        first_img = sources[0]
        W_orig, H_orig = first_img.size
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = Target_W, Target_H
    
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    tensor_list = []
    to_tensor_transform = transforms.ToTensor()

    for img_pil in sources:
        resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        img_tensor = to_tensor_transform(resized_img)
        tensor_list.append(img_tensor)

    if not tensor_list:
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Pi3 demo with viser for 3D visualization")
    parser.add_argument(
        "--input",
        type=str,
        default="data/examples/office",
        help="Path to input (folder of images or a video file)",
    )
    parser.add_argument("--input2", type=str, default=None, help="Secondary input path.")
    parser.add_argument("--input3", type=str, default=None, help="Tertiary input path.")
    parser.add_argument("--input4", type=str, default=None, help="Fourth input path.")
    parser.add_argument("--input5", type=str, default=None, help="Fifth input path.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for video processing")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for video processing (-1 for last frame)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame extraction/loading")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument("--share", action="store_true", help="Share the viser server with others")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=20.0,
        help="Initial confidence threshold (percentage)",
    )
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results_pi3",
        help="Path to folder to save inference results",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to folder or .pt file to load pre-computed inference results from",
    )
    parser.add_argument("--seq_name", type=str, default=None, help="Name of the sequence for saving results")
    parser.add_argument("--subsample", type=int, default=2, help="Subsample the point cloud for visualization by this factor")
    parser.add_argument("--video_width", type=int, default=320, help="Width of the video display in the GUI")
    parser.add_argument("--skip_viser", action="store_true", help="Skip viser visualization and only run inference")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ckpts/LoGeR_star/latest.pt",
        help="Name of the model to load from Hugging Face Hub or a local path to a checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ckpts/LoGeR_star/original_config.yaml",
        help="Path to a yaml config file for model initialization.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target resolution for input images as width and height.",
    )
    parser.add_argument("--window_size", type=int, default=32, help="Window size for non-causal inference (-1 for full sequence).")
    parser.add_argument("--overlap_size", type=int, default=3, help="Overlap size for sliding window inference.")
    parser.add_argument("--sim3", action="store_true", help="Use sim3 transformation for TTT.")
    parser.add_argument(
        "--sim3_scale_mode",
        type=str,
        default="median",
        choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"],
        help="Scale estimation mode for Sim3.",
    )
    parser.add_argument("--reset_every", type=int, default=None, help="Reset TTT / adapter state every N windows (0 disables).")
    parser.add_argument("--output_txt", type=str, default=None, help="Output trajectory txt file path.")
    parser.add_argument(
        "--se3",
        action="store_true",
        default=None,
        help="Use se3 transformation for TTT. If omitted, fallback to config value, then False.",
    )
    parser.add_argument("--no_ttt", action="store_true", help="Disable TTT.")
    parser.add_argument("--no_swa", action="store_true", help="Disable SWA.")
    parser.add_argument("--pi3x", action="store_true", help="Use Pi3X model.")
    parser.add_argument("--pi3x_metric", action="store_true", default=True, help="Use metric scaling for Pi3X (default: True).")
    parser.add_argument("--no_pi3x_metric", action="store_false", dest="pi3x_metric", help="Disable metric scaling for Pi3X.")
    parser.add_argument(
        "--canonical_first_frame",
        action="store_true",
        default=True,
        help="Use first frame as canonical frame (identity pose) for visualization.",
    )
    parser.add_argument(
        "--no_canonical_first_frame",
        action="store_false",
        dest="canonical_first_frame",
        help="Do not use first frame as canonical frame.",
    )
    parser.add_argument("--warmup", action="store_true", help="Run a warmup inference pass to trigger torch.compile before timing.")
    parser.add_argument("--benchmark", action="store_true", help="Run multiple inference passes and report timing statistics.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    input_paths = [
        Path(path).expanduser()
        for path in [args.input, args.input2, args.input3, args.input4, args.input5]
        if path is not None
    ]
    missing_inputs = [str(path) for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Input path not found: {', '.join(missing_inputs)}")

    config_path = Path(args.config).expanduser() if args.config else None
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model_name = args.model_name
    if not _is_huggingface_model_name(model_name):
        model_path = Path(model_name).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        model_name = str(model_path)

    config = _load_yaml_config(config_path)
    predictions_dict = None
    temp_frame_dirs: dict[str, Path] = {}
    all_image_names_collected: List[str] = []
    image_folder_for_sky = None
    target_resolution = list(args.resolution) if args.resolution else None

    if args.seq_name is None:
        args.seq_name = _build_sequence_name(input_paths)

    try:
        if args.load:
            load_path = _resolve_saved_predictions_path(Path(args.load).expanduser(), args.seq_name)
            if load_path.exists():
                print(f"Loading pre-computed results from {load_path}...")
                try:
                    raw_predictions = torch.load(str(load_path), map_location="cpu", weights_only=False)
                except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
                    print(f"Error loading {load_path}: {exc}. Proceeding with inference.")
                else:
                    predictions_dict = _convert_tensor_tree_to_numpy(raw_predictions)
                    image_folder_for_sky = str(input_paths[0])
                    print("Successfully loaded pre-computed results.")
            else:
                print(f"No pre-computed results found at {load_path}. Proceeding with inference.")

        if predictions_dict is None:
            model = load_pi3_model(model_name, config, args.pi3x, args.pi3x_metric).to(device).eval()
            all_image_names_collected, _, temp_frame_dirs = _collect_input_frames(
                input_paths,
                args.start_frame,
                args.end_frame,
                args.stride,
            )

            if not all_image_names_collected:
                print("No images to process. Exiting.")
                return

            print(f"Found {len(all_image_names_collected)} images to process.")
            if target_resolution is not None:
                images_tensor = load_images_from_paths(
                    all_image_names_collected,
                    Target_W=target_resolution[0],
                    Target_H=target_resolution[1],
                ).to(device)
            else:
                images_tensor = load_images_from_paths(all_image_names_collected).to(device)

            if images_tensor.numel() == 0:
                print("Error: No images were loaded successfully. Check image paths and formats.")
                return

            image_folder_for_sky = str(Path(all_image_names_collected[0]).parent)
            print("Running inference...")
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8
                else torch.float16
            )
            num_frames = images_tensor.shape[0]
            forward_kwargs = _build_forward_kwargs(args, config)
            print(f"Forward pass kwargs: {forward_kwargs}")

            if args.warmup or args.benchmark:
                print("Running warmup inference (to trigger torch.compile)...")
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    _ = model(images_tensor[None], **forward_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("Warmup complete.")

            if args.benchmark:
                num_runs = 3
                print(f"\nRunning benchmark with {num_runs} inference passes...")
                inference_times = []
                for run_idx in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_start = time.time()
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                        raw_model_predictions = model(images_tensor[None], **forward_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_end = time.time()
                    inference_times.append(t_end - t_start)
                    print(f"  Run {run_idx + 1}/{num_runs}: {t_end - t_start:.3f}s")

                avg_time = sum(inference_times) / len(inference_times)
                min_time = min(inference_times)
                max_time = max(inference_times)
                std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5

                print(f"\n{'=' * 50}")
                print(f"Benchmark Results ({num_runs} runs):")
                print(f"  Total frames: {num_frames}")
                print(f"  Avg inference time: {avg_time:.3f}s (std: {std_time:.3f}s)")
                print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
                print(f"  Avg FPS: {num_frames / avg_time:.2f}")
                print(f"  Avg time per frame: {(avg_time / num_frames) * 1000:.2f} ms")
                print(f"{'=' * 50}\n")
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_start_time = time.time()

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    raw_model_predictions = model(images_tensor[None], **forward_kwargs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_end_time = time.time()

                inference_time = inference_end_time - inference_start_time
                fps = num_frames / inference_time
                ms_per_frame = (inference_time / num_frames) * 1000
                print(f"\n{'=' * 50}")
                print("Inference Timing Results:")
                print(f"  Total frames: {num_frames}")
                print(f"  Inference time: {inference_time:.3f} seconds")
                print(f"  FPS: {fps:.2f}")
                print(f"  Time per frame: {ms_per_frame:.2f} ms")
                if not args.warmup:
                    print("  (Note: First run includes torch.compile overhead. Use --warmup for accurate timing)")
                print(f"{'=' * 50}\n")

            raw_model_predictions["images"] = images_tensor[None].permute(0, 1, 3, 4, 2)
            raw_model_predictions["conf"] = torch.sigmoid(raw_model_predictions["conf"])
            if "local_points" in raw_model_predictions:
                del raw_model_predictions["local_points"]

            predictions_dict = {
                key: value.squeeze(0).cpu().float().numpy()
                for key, value in raw_model_predictions.items()
                if value is not None and torch.is_tensor(value)
            }

            if args.output_folder:
                output_dir = Path(args.output_folder).expanduser()
                output_dir.mkdir(parents=True, exist_ok=True)
                seq_name_to_use = f"{args.seq_name}_{args.start_frame}_{args.end_frame}_{args.stride}"
                if len(input_paths) > 1:
                    seq_name_to_use += f"_x{len(input_paths)}"
                output_path = output_dir / f"{seq_name_to_use}.pt"
                print(f"Saving inference results to {output_path}...")
                torch.save(
                    {key: torch.from_numpy(value) for key, value in predictions_dict.items()},
                    output_path,
                )

        if args.output_txt and predictions_dict is not None and "camera_poses" in predictions_dict:
            output_txt_path = Path(args.output_txt).expanduser()
            print(f"Saving trajectory to {output_txt_path}...")
            fallback_frame_count = len(all_image_names_collected) or int(predictions_dict["camera_poses"].shape[0])
            if len(input_paths) == 1 and input_paths[0].is_dir():
                current_frames = _list_directory_images(
                    input_paths[0],
                    args.start_frame,
                    args.end_frame,
                    args.stride,
                )
                timestamps = _try_load_timestamps_for_images(current_frames, input_paths[0])
            else:
                timestamps = [float(index) for index in range(fallback_frame_count)]

            camera_poses = torch.from_numpy(predictions_dict["camera_poses"])
            rotation_world = camera_poses[..., :3, :3]
            translation_world = camera_poses[..., :3, 3]
            quaternion_world = mat_to_quat(rotation_world)
            sequence_length = min(len(timestamps), translation_world.shape[0], quaternion_world.shape[0])

            write_trajectory_txt(
                output_txt_path,
                timestamps[:sequence_length],
                translation_world[:sequence_length].tolist(),
                quaternion_world[:sequence_length].tolist(),
            )

        if predictions_dict is None:
            print("Error: Predictions are not available. Exiting.")
            return

        if args.skip_viser:
            print("Skipping viser visualization.")
            return

        print("Starting viser visualization...")
        viser_wrapper(
            predictions_dict,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            background_mode=args.background_mode,
            mask_sky=args.mask_sky,
            image_folder_for_sky_mask=image_folder_for_sky,
            subsample=args.subsample,
            video_width=args.video_width,
            share=args.share,
            canonical_first_frame=args.canonical_first_frame,
        )
        print("Visualization setup complete. Server is running.")
    finally:
        _cleanup_temp_dirs(temp_frame_dirs)


if __name__ == "__main__":
    main()
