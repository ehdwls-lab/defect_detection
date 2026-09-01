from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

try:
    from src.autoencoder import ConvAutoencoder
    from src.preprocessing import preprocess_anomaly
except ModuleNotFoundError:
    from autoencoder import ConvAutoencoder
    from preprocessing import preprocess_anomaly


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE_METHODS = ("mean_mse", "top1_mean", "top5_mean", "top10_mean")
TOP_FRACTIONS = {"top1_mean": 0.01, "top5_mean": 0.05, "top10_mean": 0.10}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_SUFFIXES = {"mean_mse": "mean", "top1_mean": "top1", "top5_mean": "top5", "top10_mean": "top10"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정상 검증 Patch로 score별 threshold를 계산하고 전체 이미지 결과를 생성")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "best_autoencoder.pth")
    parser.add_argument("--val-manifest", type=Path, default=PROJECT_ROOT / "data" / "manifests" / "val.csv")
    parser.add_argument("--test-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "inference")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap 합성 비율")
    parser.add_argument(
        "--score-method", choices=SCORE_METHODS, default="mean_mse",
        help="기존 단일 결과 파일에 사용할 방식; 모든 방식은 항상 계산/저장됨",
    )
    return parser.parse_args()


def load_bgr_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return image_bgr


def find_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {directory}")
    image_paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"이미지가 없습니다: {directory}")
    return image_paths


def calculate_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        raise ValueError(f"영상 크기 {length}가 Patch 크기 {patch_size}보다 작습니다.")
    return list(range(0, length - patch_size + 1, stride))


def make_patch_tensor(
    image_bgr: np.ndarray,
    patch_size: int,
    stride: int,
    preprocessing_params: dict[str, float],
) -> tuple[Tensor, list[tuple[int, int]]]:
    processed_bgr = preprocess_anomaly(image_bgr, **preprocessing_params)
    height, width = processed_bgr.shape[:2]
    patches: list[Tensor] = []
    positions: list[tuple[int, int]] = []
    for y in calculate_positions(height, patch_size, stride):
        for x in calculate_positions(width, patch_size, stride):
            patch_bgr = processed_bgr[y:y + patch_size, x:x + patch_size]
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            patch = np.ascontiguousarray(patch_rgb.transpose(2, 0, 1))
            patches.append(torch.from_numpy(patch).float() / 255.0)
            positions.append((x, y))
    return torch.stack(patches), positions


def calculate_scores(pixel_residual: Tensor) -> dict[str, Tensor]:
    flattened = pixel_residual.flatten(start_dim=1)
    scores = {"mean_mse": flattened.mean(dim=1)}
    for method, fraction in TOP_FRACTIONS.items():
        count = max(1, int(np.ceil(flattened.shape[1] * fraction)))
        scores[method] = torch.topk(flattened, k=count, dim=1).values.mean(dim=1)
    return scores


@torch.inference_mode()
def inspect_image(
    model: ConvAutoencoder,
    image_bgr: np.ndarray,
    patch_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    preprocessing_params: dict[str, float],
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    patches, positions = make_patch_tensor(image_bgr, patch_size, stride, preprocessing_params)
    height, width = image_bgr.shape[:2]
    reconstruction_sum = np.zeros((height, width, 3), dtype=np.float32)
    residual_sum = np.zeros((height, width), dtype=np.float32)
    overlap_count = np.zeros((height, width), dtype=np.float32)
    score_sums = {method: np.zeros((height, width), dtype=np.float32) for method in SCORE_METHODS}
    all_scores: dict[str, list[float]] = {method: [] for method in SCORE_METHODS}

    for start_index in range(0, len(patches), batch_size):
        end_index = min(start_index + batch_size, len(patches))
        batch = patches[start_index:end_index].to(device)
        reconstructed = model(batch)
        pixel_residual = (batch - reconstructed).square().mean(dim=1)
        batch_scores = calculate_scores(pixel_residual)
        reconstructed_numpy = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 1)
        residual_numpy = pixel_residual.detach().cpu().numpy()
        for local_index in range(end_index - start_index):
            patch_index = start_index + local_index
            x, y = positions[patch_index]
            patch_slice = np.s_[y:y + patch_size, x:x + patch_size]
            reconstruction_sum[patch_slice] += reconstructed_numpy[local_index]
            residual_sum[patch_slice] += residual_numpy[local_index]
            overlap_count[patch_slice] += 1.0
            for method in SCORE_METHODS:
                score = float(batch_scores[method][local_index].detach().cpu())
                score_sums[method][patch_slice] += score
                all_scores[method].append(score)

    safe_count = np.maximum(overlap_count, 1.0)
    reconstructed_rgb = np.clip(reconstruction_sum / safe_count[..., None], 0.0, 1.0)
    score_maps = {method: score_sums[method] / safe_count for method in SCORE_METHODS}
    residual_map = residual_sum / safe_count
    scores = {method: np.asarray(values, dtype=np.float32) for method, values in all_scores.items()}
    return scores, positions, reconstructed_rgb, score_maps, residual_map


def build_anomaly_mask(image_shape: tuple[int, int], positions: list[tuple[int, int]], scores: np.ndarray, threshold: float, patch_size: int) -> np.ndarray:
    height, width = image_shape
    votes = np.zeros((height, width), dtype=np.float32)
    total = np.zeros((height, width), dtype=np.float32)
    for (x, y), score in zip(positions, scores):
        patch_slice = np.s_[y:y + patch_size, x:x + patch_size]
        total[patch_slice] += 1.0
        if score > threshold:
            votes[patch_slice] += 1.0
    mask = (votes / np.maximum(total, 1.0) >= 0.5).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))


def make_heatmap(score_map: np.ndarray, validation_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    lower = float(np.percentile(validation_scores, 1.0))
    upper = max(float(np.percentile(test_scores, 99.0)), float(np.percentile(validation_scores, 99.0)))
    normalized = np.clip((score_map - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET)


def make_threshold_overlay(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = np.zeros((image.shape[0] + 36, image.shape[1], 3), dtype=np.uint8)
    canvas[36:, :] = image
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_validation_scores(output_path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["image", "patch_index", "x", "y", *SCORE_METHODS]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_test_scores(output_path: Path, positions: list[tuple[int, int]], scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> None:
    fields = ["patch_index", "x", "y", *SCORE_METHODS, *(f"{method}_is_anomaly" for method in SCORE_METHODS)]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, (x, y) in enumerate(positions):
            row: dict[str, object] = {"patch_index": index, "x": x, "y": y}
            for method in SCORE_METHODS:
                row[method] = float(scores[method][index])
                row[f"{method}_is_anomaly"] = int(scores[method][index] > thresholds[method])
            writer.writerow(row)


def read_validation_paths(manifest_path: Path) -> list[Path]:
    paths = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("split") == "val" and row.get("label") == "normal":
                path = Path(row["path"])
                paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    if not paths:
        raise RuntimeError(f"정상 validation image가 없습니다: {manifest_path}")
    return paths


def load_model(checkpoint: dict, device: torch.device) -> ConvAutoencoder:
    config = checkpoint["config"]
    model = ConvAutoencoder(in_channels=config["in_channels"], latent_channels=config["latent_channels"]).to(device)
    model_state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if model_state is None:
        raise KeyError("checkpoint에 model_state 또는 model_state_dict가 없습니다.")
    model.load_state_dict(model_state)
    model.eval()
    return model


@torch.inference_mode()
def main() -> None:
    args = parse_arguments()
    if not args.model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {args.model_path}")
    if not args.test_image.exists():
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {args.test_image}")
    if not 0.0 <= args.threshold_percentile <= 100.0:
        raise ValueError("threshold-percentile은 0부터 100 사이여야 합니다.")
    if args.batch_size <= 0:
        raise ValueError("batch-size는 양수여야 합니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    preprocessing_params = {
        "gamma": float(checkpoint.get("gamma", config.get("gamma", 0.82))),
        "clahe_clip": float(checkpoint.get("clahe_clip", config.get("clahe_clip", 1.5))),
        "unsharp_amount": float(checkpoint.get("unsharp_amount", config.get("unsharp_amount", 0.30))),
    }
    patch_size = int(checkpoint.get("patch_size", config["patch_size"]))
    stride = int(checkpoint.get("stride", config["stride"]))
    model = load_model(checkpoint, device)

    validation_scores: dict[str, list[float]] = {method: [] for method in SCORE_METHODS}
    validation_rows: list[dict[str, object]] = []
    validation_paths = read_validation_paths(args.val_manifest)
    for validation_path in validation_paths:
        scores, positions, _, _, _ = inspect_image(model, load_bgr_image(validation_path), patch_size, stride, args.batch_size, device, preprocessing_params)
        for index, (x, y) in enumerate(positions):
            row: dict[str, object] = {"image": validation_path.name, "patch_index": index, "x": x, "y": y}
            for method in SCORE_METHODS:
                value = float(scores[method][index])
                validation_scores[method].append(value)
                row[method] = value
            validation_rows.append(row)

    validation_arrays = {method: np.asarray(values, dtype=np.float32) for method, values in validation_scores.items()}
    thresholds = {method: float(np.percentile(values, args.threshold_percentile)) for method, values in validation_arrays.items()}
    test_bgr = load_bgr_image(args.test_image)
    preprocessed_bgr = preprocess_anomaly(test_bgr, **preprocessing_params)
    test_scores, test_positions, reconstructed_rgb, score_maps, residual_map = inspect_image(
        model, test_bgr, patch_size, stride, args.batch_size, device, preprocessing_params
    )
    if test_bgr.shape[:2] == (800, 1280) and len(test_positions) != 936:
        raise RuntimeError(f"1280x800 이미지의 Patch 수가 936이 아닙니다: {len(test_positions)}")

    result_dir = args.output_dir / args.test_image.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    original_bgr = test_bgr
    reconstructed_bgr = cv2.cvtColor((reconstructed_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    masks: dict[str, np.ndarray] = {}
    overlays: dict[str, np.ndarray] = {}
    heatmaps: dict[str, np.ndarray] = {}
    for method in SCORE_METHODS:
        masks[method] = build_anomaly_mask(test_bgr.shape[:2], test_positions, test_scores[method], thresholds[method], patch_size)
        overlays[method] = make_threshold_overlay(original_bgr, masks[method])
        heatmaps[method] = make_heatmap(score_maps[method], validation_arrays[method], test_scores[method])
        short_name = OUTPUT_SUFFIXES[method]
        cv2.imwrite(str(result_dir / f"anomaly_heatmap_{short_name}.png"), heatmaps[method])
        cv2.imwrite(str(result_dir / f"anomaly_mask_{short_name}.png"), masks[method])
        cv2.imwrite(str(result_dir / f"anomaly_overlay_{short_name}.png"), overlays[method])

    residual_heatmap = make_heatmap(residual_map, residual_map.ravel(), residual_map.ravel())
    residual_overlay = cv2.addWeighted(original_bgr, 1.0 - args.alpha, residual_heatmap, args.alpha, 0.0)
    comparison = np.hstack([
        add_label(original_bgr, "ORIGINAL"), add_label(reconstructed_bgr, "RECONSTRUCTION"),
        add_label(heatmaps["mean_mse"], "MEAN MSE HEATMAP"), add_label(heatmaps["top5_mean"], "TOP 5% HEATMAP"),
        add_label(overlays["mean_mse"], "MEAN THRESHOLD"), add_label(overlays["top5_mean"], "TOP 5% THRESHOLD"),
    ])
    cv2.imwrite(str(result_dir / "original.png"), original_bgr)
    cv2.imwrite(str(result_dir / "preprocessed.png"), preprocessed_bgr)
    cv2.imwrite(str(result_dir / "reconstruction.png"), reconstructed_bgr)
    cv2.imwrite(str(result_dir / "pixel_residual_map.png"), residual_heatmap)
    cv2.imwrite(str(result_dir / "pixel_residual_overlay.png"), residual_overlay)
    cv2.imwrite(str(result_dir / "comparison_scores.png"), comparison)
    cv2.imwrite(str(result_dir / "anomaly_heatmap.png"), heatmaps[args.score_method])
    cv2.imwrite(str(result_dir / "anomaly_mask.png"), masks[args.score_method])
    cv2.imwrite(str(result_dir / "anomaly_overlay.png"), overlays[args.score_method])
    cv2.imwrite(str(result_dir / "comparison.png"), comparison)
    save_validation_scores(result_dir / "validation_patch_scores.csv", validation_rows)
    save_test_scores(result_dir / "test_patch_scores.csv", test_positions, test_scores, thresholds)

    threshold_lines = [f"model={args.model_path}", f"epoch={checkpoint['epoch']}", f"patch_size={patch_size}", f"stride={stride}"]
    threshold_lines.extend(f"{key}={value}" for key, value in preprocessing_params.items())
    threshold_lines.append(f"percentile={args.threshold_percentile}")
    for method in SCORE_METHODS:
        threshold_lines.extend([
            f"[{method}]", f"validation_mean={validation_arrays[method].mean():.10f}",
            f"validation_max={validation_arrays[method].max():.10f}", f"threshold={thresholds[method]:.10f}",
            f"test_mean={test_scores[method].mean():.10f}", f"test_max={test_scores[method].max():.10f}",
            f"test_anomaly_patches={int(np.sum(test_scores[method] > thresholds[method]))}/{len(test_scores[method])}",
        ])
    (result_dir / "thresholds.txt").write_text("\n".join(threshold_lines) + "\n", encoding="utf-8")
    (result_dir / "threshold.txt").write_text("\n".join(threshold_lines) + "\n", encoding="utf-8")

    print("=" * 70)
    print(f"사용 장치: {device} | 모델 Epoch: {checkpoint['epoch']} | Patch: {len(test_positions)}")
    print(f"결과 저장: {result_dir}")
    for method in SCORE_METHODS:
        values = test_scores[method]
        print(f"\n[{method}]")
        print(f"validation mean: {validation_arrays[method].mean():.6f}")
        print(f"validation max: {validation_arrays[method].max():.6f}")
        print(f"{args.threshold_percentile:g} percentile threshold: {thresholds[method]:.6f}")
        print(f"test mean: {values.mean():.6f}")
        print(f"test max: {values.max():.6f}")
        print(f"anomaly patches / total patches: {int(np.sum(values > thresholds[method]))} / {len(values)}")
        print(f"[{method} top patches]")
        print("rank, x, y, score, threshold")
        for rank, index in enumerate(np.argsort(values)[-10:][::-1], 1):
            x, y = test_positions[index]
            print(f"{rank}, {x}, {y}, {values[index]:.10f}, {thresholds[method]:.10f}")
    print(f"legacy output score method: {args.score_method}")
    print("=" * 70)


if __name__ == "__main__":
    main()