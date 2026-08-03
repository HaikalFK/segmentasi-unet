#!/usr/bin/env python
"""
Compare Standard BCE vs Weighted BCE + Dice Loss on U-Net.

Trains two models with identical hyperparameters, different loss functions.
Saves minimal checkpoints and generates comparison artifacts.

Usage:
    python experiments/compare_losses.py
    python experiments/compare_losses.py --config-dir experiments/configs --output-dir results
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import yaml
from tqdm import tqdm
from datetime import datetime

from unet.unet_model import UNet
from utils.loss_functions import get_loss_fn
from utils.checkpoint import MinimalCheckpoint, ComparisonCheckpoint
from utils.dice_score import dice_coeff
from utils.wandb_tracker import WandBTracker


# Binary Leaf Dataset (copied from notebook for standalone script)
from PIL import Image
from torch.utils.data import Dataset


class BinaryLeafDataset(Dataset):
    """Binary leaf segmentation dataset (leaf=1, background=0)."""

    def __init__(self, img_dir, mask_dir, scale=0.5, target_size=(256, 256)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.target_size = target_size
        self.ids = [p.stem for p in Path(img_dir).glob("*.png")]
        print(f"Dataset: {len(self.ids)} samples, binary (bg + leaf)")

    def __len__(self):
        return len(self.ids)

    def preprocess_img(self, pil_img):
        w, h = pil_img.size
        tw, th = self.target_size
        pil_img = pil_img.resize((tw, th), Image.BICUBIC)
        img = np.asarray(pil_img)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        img = img / 255.0
        return img

    def preprocess_mask(self, pil_img):
        tw, th = self.target_size
        pil_img = pil_img.resize((tw, th), Image.NEAREST)
        mask = np.asarray(pil_img, dtype=np.int64)
        binary = (mask > 0).astype(np.int64)
        return binary

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(self.img_dir / f"{name}.png").convert("RGB")
        mask = Image.open(self.mask_dir / f"{name}.png")
        return {
            "image": torch.from_numpy(self.preprocess_img(img)).float(),
            "mask": torch.from_numpy(self.preprocess_mask(mask)).long(),
            "name": name,
        }


def binary_metrics(pred_mask, true_mask, eps=1e-8):
    """Compute IoU and Dice for binary masks."""
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    inter = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()
    iou = inter / (union + eps)
    dice = (2 * inter) / (pred_mask.sum() + true_mask.sum() + eps)
    return iou.item(), dice.item()


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _num(value, cast=float):
    """Coerce a YAML value to a numeric type.

    PyYAML can parse values like ``1e-4`` as str on some setups; this helper
    forces them to float (or int) so they never reach torch as strings.
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        return value


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create U-Net model from config."""
    model = UNet(
        n_channels=_num(config["model"]["n_channels"], int),
        n_classes=_num(config["model"]["n_classes"], int),
        bilinear=config["model"]["bilinear"],
    )
    model = model.to(memory_format=torch.channels_last)
    model.to(device)
    return model


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from config."""
    train_cfg = config["training"]
    lr = _num(train_cfg["learning_rate"])
    weight_decay = _num(train_cfg["weight_decay"])
    momentum = _num(train_cfg["momentum"])
    if train_cfg["optimizer"] == "RMSprop":
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=False,
            eps=1e-8,
        )
    elif train_cfg["optimizer"] == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif train_cfg["optimizer"] == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create LR scheduler from config."""
    train_cfg = config["training"]
    if train_cfg["scheduler"] == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=_num(train_cfg["scheduler_patience"], int),
            min_lr=_num(train_cfg["scheduler_min_lr"]),
        )
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: GradScaler,
    amp: bool,
    gradient_clip: float,
):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in loader:
        imgs = batch["image"].to(device, dtype=torch.float32, memory_format=torch.channels_last)
        masks = batch["mask"].to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            pred = model(imgs)
            loss = loss_fn(pred, masks)

        # Skip NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            scaler.step(optimizer)
            epoch_loss += loss.item()
            num_batches += 1

        scaler.update()

    return epoch_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    amp: bool,
    return_samples: bool = False,
    max_samples: int = 3,
):
    """Validate model.

    Args:
        return_samples: if True, also return sample images/masks/preds for W&B logging.
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    samples = []  # list of (image_chw_float, true_mask_hw, pred_mask_hw)

    for batch in loader:
        imgs = batch["image"].to(device, dtype=torch.float32, memory_format=torch.channels_last)
        masks = batch["mask"].to(device, dtype=torch.long)

        with autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            pred = model(imgs)
            loss = loss_fn(pred, masks)

        total_loss += loss.item()

        pred_bin = pred.argmax(dim=1)
        iou, dice = binary_metrics(pred_bin, masks)
        total_iou += iou
        total_dice += dice
        num_batches += 1

        # Collect sample predictions (only first batch, limited count)
        if return_samples and not samples:
            for k in range(min(imgs.shape[0], max_samples)):
                img = imgs[k].cpu()                       # (C, H, W) float in [0,1]
                true = masks[k].cpu().float()             # (H, W)
                predm = pred_bin[k].cpu().float()         # (H, W)
                samples.append((img, true, predm))

    metrics = {
        "val_loss": total_loss / max(num_batches, 1),
        "val_iou": total_iou / max(num_batches, 1),
        "val_dice": total_dice / max(num_batches, 1),
    }
    if return_samples:
        metrics["samples"] = samples
    return metrics


def run_experiment(config_path: Path, device: torch.device) -> dict:
    """Run full training for one loss function."""
    config = load_config(config_path)
    loss_name = config["loss"]["name"]

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {loss_name}")
    print(f"{'='*60}")

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Data
    data_cfg = config["data"]
    dataset = BinaryLeafDataset(
        data_cfg["img_dir"],
        data_cfg["mask_dir"],
        scale=_num(data_cfg["scale"]),
        target_size=tuple(_num(v, int) for v in data_cfg["target_size"]),
    )

    n_val = int(len(dataset) * _num(data_cfg["validation_pct"]) / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config["experiment"]["seed"]))

    loader_args = dict(
        batch_size=_num(config["training"]["batch_size"], int),
        num_workers=_num(data_cfg["num_workers"], int),
        pin_memory=data_cfg["pin_memory"],
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    print(f"Train: {n_train}  Val: {n_val}")

    # Model, optimizer, scheduler
    model = create_model(config, device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler(device.type, enabled=bool(config["training"]["amp"]))

    # Loss function
    loss_cfg = config["loss"]
    loss_fn = get_loss_fn(loss_cfg["name"])
    if loss_cfg["name"] == "weighted_bce_dice":
        # Wrap with parameters
        def wrapped_loss(pred, target):
            clamp = tuple(_num(v) for v in loss_cfg.get("weight_clamp", (0.3, 3.0)))
            return loss_fn(
                pred,
                target,
                weight_clamp=clamp,
                dice_weight=_num(loss_cfg.get("dice_weight", 1.0)),
            )
        loss_fn = wrapped_loss

    # Checkpoint manager
    output_cfg = config["output"]
    save_dir = Path(output_cfg["save_dir"])
    checkpoint = MinimalCheckpoint(save_dir, experiment_name=config["experiment"]["name"])

    # Training configuration (assigned before tracker.init uses it)
    train_cfg = config["training"]

    # W&B tracker (graceful no-op if disabled/missing)
    wandb_cfg = config.get("wandb", {})
    tracker = WandBTracker(wandb_cfg, loss_name=loss_name)
    tracker.init(extra_config={"loss": loss_name, "training": train_cfg, "data": data_cfg})

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_dice = -1.0
    best_state = None
    stale_epochs = 0

    epochs = _num(train_cfg["epochs"], int)
    patience = _num(train_cfg["early_stop_patience"], int)
    delta = _num(train_cfg["early_stop_delta"])
    log_interval = _num(wandb_cfg.get("log_interval", 1), int)
    log_samples = bool(wandb_cfg.get("log_sample_predictions", True))
    grad_clip = _num(train_cfg["gradient_clip"])

    print(f"Epochs: {epochs}, Early stop patience: {patience}, Delta: {delta}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler, train_cfg["amp"], grad_clip
        )

        # Validate — capture samples only when we'll log them
        val_metrics = validate(
            model, val_loader, loss_fn, device, train_cfg["amp"],
            return_samples=(log_samples and epoch % log_interval == 0),
        )

        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_metrics["val_dice"])

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_iou"].append(val_metrics["val_iou"])
        history["val_dice"].append(val_metrics["val_dice"])

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | IoU: {val_metrics['val_iou']:.4f} | Dice: {val_metrics['val_dice']:.4f}")

        # W&B logging every log_interval epochs
        if epoch % log_interval == 0:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_iou": val_metrics["val_iou"],
                "val_dice": val_metrics["val_dice"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            # Sample prediction images
            if log_samples and val_metrics.get("samples"):
                for si, (img, true, predm) in enumerate(val_metrics["samples"]):
                    log_data[f"sample_{si}/input"] = tracker.image(img)
                    log_data[f"sample_{si}/truth"] = tracker.image(true.unsqueeze(0))
                    log_data[f"sample_{si}/prediction"] = tracker.image(predm.unsqueeze(0))
            tracker.log(log_data)

        # Checkpoint best
        if val_metrics["val_dice"] > best_dice + delta:
            best_dice = val_metrics["val_dice"]
            best_state = model.state_dict()
            stale_epochs = 0
            print(f"  ★ New best Dice: {best_dice:.4f}")
        else:
            stale_epochs += 1

        # Early stopping (inside training loop)
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Finish W&B run
    tracker.finish()

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint.save_best(
        model=model,
        config=config,
        metrics_history=history,
        loss_name=loss_name,
        best_metric_value=best_dice,
        best_metric_name="val_dice",
    )

    return {
        "loss_name": loss_name,
        "config": config,
        "history": history,
        "best_dice": best_dice,
        "save_dir": save_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare loss functions for U-Net")
    parser.add_argument("--config-dir", type=str, default="experiments/configs", help="Directory with config YAMLs")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--losses", nargs="+", default=["standard_bce", "weighted_bce_dice"], help="Loss functions to compare")
    parser.add_argument("--cleanup", action="store_true", help="Delete .pth files after comparison")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run experiments
    results = {}
    for loss_name in args.losses:
        config_path = config_dir / f"{loss_name}.yaml"
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue
        result = run_experiment(config_path, device)
        results[loss_name] = result

    # Generate comparison
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON ARTIFACTS")
    print(f"{'='*60}")

    comparison = ComparisonCheckpoint(output_dir)

    # Aggregate metrics
    aggregated = {}
    for loss_name, result in results.items():
        metrics_path = Path(result["save_dir"]) / f"metrics_{loss_name}.json"
        if metrics_path.exists():
            aggregated[loss_name] = {
                "metrics": MinimalCheckpoint.load_metrics(metrics_path),
                "config": result["config"],
            }

    # Save comparison CSV & MD
    comparison.save_comparison_csv(aggregated)
    comparison.save_comparison_md(aggregated)

    # Generate plots
    comparison.plot_comparison(aggregated)

    # Optional cleanup
    if args.cleanup:
        print("\nCleaning up model files...")
        for loss_name in args.losses:
            save_dir = output_dir / loss_name
            MinimalCheckpoint.cleanup_all_models(save_dir)

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Results in: {output_dir}")
    print(f"Comparison in: {output_dir}/comparison")


if __name__ == "__main__":
    main()