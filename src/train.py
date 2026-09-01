from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from src.autoencoder import ConvAutoencoder
    from src.patch_dataset import PatchDataset
except ModuleNotFoundError:
    from autoencoder import ConvAutoencoder
    from patch_dataset import PatchDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정상 RGB Patch 기반 Autoencoder 학습"
    )

    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "train.csv",
        help="정상 학습 manifest CSV",
    )

    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "val.csv",
        help="정상 검증 manifest CSV",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="모델 저장 폴더",
    )

    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--latent-channels", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.82)
    parser.add_argument("--clahe-clip", type=float, default=1.5)
    parser.add_argument("--unsharp-amount", type=float, default=0.30)
    parser.add_argument(
        "--allow-full-image",
        action="store_true",
        help="region JSON이 없는 이미지의 전체 영역 학습을 명시적으로 허용",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    import csv

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def validate_manifests(train_manifest: Path, val_manifest: Path) -> None:
    train_rows = read_manifest_rows(train_manifest)
    val_rows = read_manifest_rows(val_manifest)
    train_paths = {row["path"] for row in train_rows}
    val_paths = {row["path"] for row in val_rows}
    if train_paths & val_paths:
        raise ValueError("train과 val source image가 중복됩니다.")
    if any(row.get("split") != "train" or row.get("label") != "normal" for row in train_rows):
        raise ValueError("train manifest에는 split=train, label=normal만 허용됩니다.")
    if any(row.get("split") != "val" or row.get("label") != "normal" for row in val_rows):
        raise ValueError("val manifest에는 split=val, label=normal만 허용됩니다.")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for patches in data_loader:
        patches: Tensor = patches.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        reconstructed = model(patches)
        loss = criterion(reconstructed, patches)

        loss.backward()
        optimizer.step()

        batch_size = patches.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.inference_mode()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for patches in data_loader:
        patches: Tensor = patches.to(
            device,
            non_blocking=True,
        )

        reconstructed = model(patches)
        loss = criterion(reconstructed, patches)

        batch_size = patches.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def main() -> None:
    args = parse_arguments()
    set_seed(args.seed)

    if args.patch_size != 64 or args.stride != 32:
        raise ValueError(
            "첫 파일럿 설정은 patch-size=64, stride=32여야 합니다."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"사용 장치: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    if not args.train_manifest.exists() or not args.val_manifest.exists():
        raise FileNotFoundError("train/val manifest를 먼저 생성해야 합니다.")
    validate_manifests(args.train_manifest, args.val_manifest)

    # 위의 if/else와 관계없이 Dataset 생성
    train_dataset = PatchDataset(
        manifest_path=args.train_manifest,
        split="train",
        patch_size=args.patch_size,
        stride=args.stride,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        unsharp_amount=args.unsharp_amount,
        allow_full_image=args.allow_full_image,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        f"학습 Patch 수: "
        f"{len(train_dataset)}"
    )

    val_dataset = PatchDataset(
        manifest_path=args.val_manifest,
        split="val",
        patch_size=args.patch_size,
        stride=args.stride,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        unsharp_amount=args.unsharp_amount,
        allow_full_image=args.allow_full_image,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"검증 Patch 수: {len(val_dataset)}")

    model = ConvAutoencoder(
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
    ).to(device)

    # 입력 Patch와 복원 Patch의 평균제곱오차
    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        args.output_dir
        / "best_autoencoder.pth"
    )

    last_model_path = (
        args.output_dir
        / "last_autoencoder.pth"
    )

    best_loss = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        if val_loader is not None:
            val_loss = validate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            comparison_loss = val_loss

            print(
                f"Epoch "
                f"[{epoch:03d}/{args.epochs:03d}] "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        else:
            val_loss = None
            comparison_loss = val_loss

            print(
                f"Epoch "
                f"[{epoch:03d}/{args.epochs:03d}] "
                f"Train Loss: {train_loss:.6f}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": {
                "patch_size": (
                    args.patch_size
                ),
                "stride": args.stride,
                "in_channels": (
                    args.in_channels
                ),
                "latent_channels": (
                    args.latent_channels
                ),
                "gamma": args.gamma,
                "clahe_clip": args.clahe_clip,
                "unsharp_amount": args.unsharp_amount,
            },
        }

        # 현재까지 가장 낮은 검증 Loss 모델 저장
        if comparison_loss < best_loss:
            best_loss = comparison_loss

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                "  → Best 모델 저장: "
                f"{best_model_path}"
            )

        # 마지막 Epoch 상태도 계속 저장
        torch.save(
            checkpoint,
            last_model_path,
        )

    print("\n학습 완료")
    print(
        f"Best 모델: "
        f"{best_model_path}"
    )
    print(
        f"Last 모델: "
        f"{last_model_path}"
    )


if __name__ == "__main__":
    main()