from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ConvAutoencoder
from src.depth_roi import (
    calculate_patch_coverage,
    create_object_mask,
    make_depth_visualization,
    make_mask_views,
)
from src.infer_anomaly import calculate_positions


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth Object Mask 내부 패치만 기존 Autoencoder로 검사"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="color.png와 depth.npy가 있는 sample 폴더")
    parser.add_argument("--model", type=Path, default=(
        PROJECT_ROOT / "results" / "pilot_v1" / "models" / "best_autoencoder.pth"
    ))
    parser.add_argument("--threshold", type=float, default=0.0001805242)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-height-mm", type=float, default=5.0)
    parser.add_argument("--border-ratio", type=float, default=0.10)
    parser.add_argument("--min-object-area", type=int, default=10000)
    parser.add_argument("--mask-coverage", type=float, default=0.90)
    parser.add_argument("--mask-erode", type=int, default=3)
    parser.add_argument("--morphology-size", type=int, default=5)
    parser.add_argument("--all-components", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConvAutoencoder(
        in_channels=int(config["in_channels"]),
        latent_channels=int(config["latent_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, int(config["patch_size"]), int(config["stride"])


def select_patch_tensors(
    color_rgb: np.ndarray,
    object_mask: np.ndarray,
    patch_size: int,
    stride: int,
    required_coverage: float,
):
    height, width = color_rgb.shape[:2]
    positions = [
        (x, y)
        for y in calculate_positions(height, patch_size, stride)
        for x in calculate_positions(width, patch_size, stride)
    ]
    coverages = calculate_patch_coverage(object_mask, positions, patch_size)
    selected_indexes = np.flatnonzero(coverages >= required_coverage)
    tensors = []
    selected_positions = []
    for index in selected_indexes:
        x, y = positions[int(index)]
        patch = np.ascontiguousarray(
            color_rgb[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1)
        )
        tensors.append(torch.from_numpy(patch).float() / 255.0)
        selected_positions.append((x, y))
    if not tensors:
        raise RuntimeError("Mask coverage 조건을 통과한 Object Patch가 없습니다.")
    return torch.stack(tensors), positions, coverages, selected_positions


@torch.inference_mode()
def inspect_selected_patches(
    model, patches, positions, image_shape, patch_size, batch_size, device
):
    height, width = image_shape
    score_sum = np.zeros((height, width), np.float32)
    reconstruction_sum = np.zeros((height, width, 3), np.float32)
    count_map = np.zeros((height, width), np.float32)
    scores = []
    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size].to(device)
        reconstructed = model(batch)
        batch_scores = ((batch - reconstructed) ** 2).mean((1, 2, 3)).cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy().transpose(0, 2, 3, 1)
        for local_index, score in enumerate(batch_scores):
            index = start + local_index
            x, y = positions[index]
            score_sum[y:y + patch_size, x:x + patch_size] += float(score)
            reconstruction_sum[y:y + patch_size, x:x + patch_size] += reconstructed_np[local_index]
            count_map[y:y + patch_size, x:x + patch_size] += 1
            scores.append(float(score))
    valid_pixels = count_map > 0
    score_map = np.zeros_like(score_sum)
    score_map[valid_pixels] = score_sum[valid_pixels] / count_map[valid_pixels]
    reconstruction = np.zeros_like(reconstruction_sum)
    reconstruction[valid_pixels] = (
        reconstruction_sum[valid_pixels] / count_map[valid_pixels, None]
    )
    return np.asarray(scores, np.float32), score_map, reconstruction, valid_pixels


def build_selected_anomaly_mask(
    image_shape, positions, scores, threshold, patch_size, object_mask
):
    votes = np.zeros(image_shape, np.float32)
    totals = np.zeros(image_shape, np.float32)
    for (x, y), score in zip(positions, scores):
        totals[y:y + patch_size, x:x + patch_size] += 1
        if score > threshold:
            votes[y:y + patch_size, x:x + patch_size] += 1
    mask = ((votes / np.maximum(totals, 1)) >= 0.5) & (object_mask > 0)
    return mask.astype(np.uint8) * 255


def main() -> None:
    args = parse_arguments()
    if not 0.0 <= args.mask_coverage <= 1.0:
        raise ValueError("mask-coverage는 0~1 범위여야 합니다.")
    if args.batch_size <= 0 or args.threshold < 0:
        raise ValueError("batch-size는 양수, threshold는 0 이상이어야 합니다.")
    color_path = args.input / "color.png"
    depth_path = args.input / "depth.npy"
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"RGB를 읽을 수 없습니다: {color_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth가 없습니다: {depth_path}")
    depth_mm = np.load(depth_path, allow_pickle=False)
    if depth_mm.shape != color_bgr.shape[:2]:
        raise ValueError(
            "RGB와 aligned Depth 크기가 다릅니다. Resize하지 않습니다: "
            f"RGB={color_bgr.shape[:2]}, Depth={depth_mm.shape}"
        )
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mask_result = create_object_mask(
        depth_mm, args.min_height_mm, args.border_ratio,
        args.min_object_area, not args.all_components,
        args.morphology_size, args.mask_erode,
    )
    if mask_result.mask_area == 0:
        raise RuntimeError("Object Mask가 비어 있습니다. Depth와 파라미터를 확인하십시오.")
    if mask_result.invalid_ratio_in_bbox > 0.20:
        print("[WARNING] Object candidate region contains "
              f"{mask_result.invalid_ratio_in_bbox:.1%} invalid depth pixels. "
              "Depth ROI may be unreliable.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, patch_size, stride = load_model(args.model, device)
    if patch_size != 64 or stride != 32:
        raise RuntimeError(f"기존 요구 설정과 다릅니다: patch={patch_size}, stride={stride}")
    patches, all_positions, coverages, valid_positions = select_patch_tensors(
        color_rgb, mask_result.mask, patch_size, stride, args.mask_coverage
    )
    scores, score_map, reconstruction, valid_pixels = inspect_selected_patches(
        model, patches, valid_positions, color_rgb.shape[:2],
        patch_size, args.batch_size, device,
    )
    anomaly_mask = build_selected_anomaly_mask(
        color_rgb.shape[:2], valid_positions, scores, args.threshold,
        patch_size, mask_result.mask,
    )
    anomaly_count = int(np.sum(scores > args.threshold))
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "depth_roi_test" / args.input.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_vis = make_depth_visualization(depth_mm)
    mask_overlay, bbox_preview, masked_rgb = make_mask_views(
        color_bgr, mask_result.mask, mask_result.bbox
    )
    heatmap_upper = max(float(np.percentile(scores, 99)), args.threshold)
    normalized = np.clip(score_map / max(heatmap_upper, 1e-12), 0, 1)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap[~valid_pixels] = 0
    overlay = color_bgr.copy()
    blended = cv2.addWeighted(color_bgr, 1 - args.alpha, heatmap, args.alpha, 0)
    overlay[valid_pixels] = blended[valid_pixels]
    threshold_result = color_bgr.copy()
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(threshold_result, contours, -1, (0, 0, 255), 2)
    reconstruction_bgr = np.zeros_like(color_bgr)
    reconstruction_u8 = cv2.cvtColor(
        (np.clip(reconstruction, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    reconstruction_bgr[valid_pixels] = reconstruction_u8[valid_pixels]
    images = {
        "original_rgb.png": color_bgr,
        "depth_visualization.png": depth_vis,
        "object_mask.png": mask_result.mask,
        "mask_overlay.png": mask_overlay,
        "roi_bbox_preview.png": bbox_preview,
        "masked_rgb.png": masked_rgb,
        "reconstruction.png": reconstruction_bgr,
        "anomaly_heatmap.png": heatmap,
        "anomaly_overlay.png": overlay,
        "anomaly_mask.png": anomaly_mask,
        "threshold_result.png": threshold_result,
    }
    for name, image in images.items():
        if not cv2.imwrite(str(output_dir / name), image):
            raise RuntimeError(f"저장 실패: {output_dir / name}")

    with (output_dir / "patch_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["patch_index", "x", "y", "mask_coverage", "mse", "is_anomaly"])
        for index, ((x, y), score) in enumerate(zip(valid_positions, scores)):
            all_index = all_positions.index((x, y))
            writer.writerow([index, x, y, float(coverages[all_index]),
                             float(score), int(score > args.threshold)])
    stats = {
        "model": str(args.model), "model_epoch": checkpoint["epoch"],
        "threshold": args.threshold, "threshold_condition": "patch_mse > threshold",
        "image_width": color_bgr.shape[1], "image_height": color_bgr.shape[0],
        "patch_size": patch_size, "stride": stride,
        "mask_coverage": args.mask_coverage,
        "all_patch_count": len(all_positions),
        "valid_object_patch_count": len(valid_positions),
        "ignored_patch_count": len(all_positions) - len(valid_positions),
        "anomaly_patch_count": anomaly_count,
        "anomaly_patch_ratio": anomaly_count / len(valid_positions),
        "mean_patch_mse": float(np.mean(scores)),
        "max_patch_mse": float(np.max(scores)),
        "floor_depth_mm": mask_result.floor_depth_mm,
        "mask_area": mask_result.mask_area,
        "bbox": mask_result.bbox,
        "invalid_depth_ratio_in_bbox": mask_result.invalid_ratio_in_bbox,
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
