"""
train_2class.py — Binary Leaf Segmentation Training

Train U-Net for binary segmentation (background vs leaf) on the
Plant Phenotyping Dataset.

This file is a self-contained training script that:
  1. Converts multi-class instance masks to binary on-the-fly
  2. Uses weighted BCE + Dice loss to handle class imbalance
  3. Saves best checkpoint based on validation Dice score
  4. Logs training history for later visualization

Usage:
    python train_2class.py
    python train_2class.py --epochs 50 --batch-size 8 --amp
    python train_2class.py --lr 5e-5 --early-stop-patience 20

Author: auto-generated from Unet_Binary_Leaf_Segmentation.ipynb
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DIR_IMG = Path('./data/imgs/')
DIR_MASK = Path('./data/masks/')
DIR_CKPT = Path('./checkpoints/')


# ===========================================================================
# 1. Binary Dataset
# ===========================================================================

class BinaryLeafDataset(Dataset):
    """
    Converts multi-class instance masks (pixel values 0..27) to binary masks
    on-the-fly: background=0, leaf=1.

    Args:
        img_dir: Path to RGB images directory
        mask_dir: Path to mask PNGs directory
        target_size: (H, W) for resizing
    """

    def __init__(self, img_dir: str, mask_dir: str, target_size=(256, 256)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size
        self.ids = sorted([p.stem for p in Path(img_dir).glob('*.png')])

        if not self.ids:
            raise RuntimeError(f'No PNG files found in {img_dir}')

        logging.info(f'BinaryLeafDataset: {len(self.ids)} samples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _preprocess_img(pil_img, target_size):
        pil_img = pil_img.resize(target_size, Image.BICUBIC)
        arr = np.asarray(pil_img)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        else:
            arr = arr.transpose((2, 0, 1))
        arr = arr / 255.0
        return arr

    @staticmethod
    def _preprocess_mask(pil_img, target_size):
        pil_img = pil_img.resize(target_size, Image.NEAREST)
        mask = np.asarray(pil_img, dtype=np.int64)
        return (mask > 0).astype(np.int64)          # binary mapping

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(self.img_dir / f'{name}.png').convert('RGB')
        mask = Image.open(self.mask_dir / f'{name}.png')
        return {
            'image': torch.from_numpy(self._preprocess_img(img, self.target_size)).float(),
            'mask': torch.from_numpy(self._preprocess_mask(mask, self.target_size)).long(),
            'name': name,
        }


# ===========================================================================
# 2. Loss Function — Weighted BCE + Dice
# ===========================================================================

def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Weighted CrossEntropy + Dice loss for imbalanced binary segmentation.

    Class weights are computed per-batch as inverse frequency to
    prevent the model from collapsing to all-background predictions.

    Args:
        pred:  logits (B, 2, H, W)
        target: ground truth (B, H, W) in {0, 1}
    Returns:
        scalar loss tensor
    """
    n_fg = (target == 1).sum().float()
    n_bg = (target == 0).sum().float()
    total = n_fg + n_bg + 1e-6

    w_fg = (total / (2 * n_fg + 1e-6)).clamp(0.3, 3.0)
    w_bg = (total / (2 * n_bg + 1e-6)).clamp(0.3, 3.0)
    weight = torch.tensor([w_bg, w_fg], device=pred.device)

    bce = F.cross_entropy(pred, target, weight=weight)
    prob = F.softmax(pred, dim=1)[:, 1]
    dice = dice_loss(prob, target.float(), multiclass=False)
    return bce + dice


# ===========================================================================
# 3. Metrics
# ===========================================================================

def binary_metrics(pred: torch.Tensor, true: torch.Tensor, eps=1e-8):
    """
    Compute binary IoU and Dice for a batch.

    Args:
        pred:  (B, H, W) in {0, 1}
        true:  (B, H, W) in {0, 1}
    Returns:
        (iou, dice) as Python floats
    """
    inter = (pred & true).sum().float()
    union = (pred | true).sum().float()
    iou = inter / (union + eps)
    dice = (2 * inter) / (pred.sum() + true.sum() + eps)
    return iou.item(), dice.item()


# ===========================================================================
# 4. Training Function
# ===========================================================================

def train_model(
    model: nn.Module,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    amp: bool = True,
    early_stop_patience: int = 15,
    early_stop_delta: float = 0.005,
    save_checkpoint: bool = True,
):
    # ── Dataset ──
    dataset = BinaryLeafDataset(str(DIR_IMG), str(DIR_MASK), target_size=(256, 256))

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    num_workers = 0 if device.type == 'cpu' else os.cpu_count()
    pin_memory = device.type == 'cuda'
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # ── Optimiser & scheduler ──
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-8,
        momentum=0.999,
        foreach=False,
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # ── Training state ──
    best_dice = -1.0
    best_state = None
    stale_epochs = 0
    global_step = 0
    history = {'epoch': [], 'train_loss': [], 'val_iou': [], 'val_dice': []}

    if save_checkpoint:
        DIR_CKPT.mkdir(parents=True, exist_ok=True)

    logging.info(
        f'Starting binary training:\n'
        f'  Epochs:          {epochs}\n'
        f'  Batch size:      {batch_size}\n'
        f'  Learning rate:   {learning_rate}\n'
        f'  Training size:   {n_train}\n'
        f'  Validation size: {n_val}\n'
        f'  Device:          {device.type}\n'
        f'  Mixed Precision: {amp}\n'
        f'  Patience:        {early_stop_patience}\n'
    )

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit='batch')

        for batch in pbar:
            imgs = batch['image'].to(
                device, dtype=torch.float32, memory_format=torch.channels_last
            )
            masks = batch['mask'].to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device.type if device.type != 'mps' else 'cpu', enabled=amp
            ):
                pred = model(imgs)
                loss = combined_loss(pred, masks)

            # Skip batch if loss is NaN (AMP instability safeguard)
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix(loss='NaN (skip)')
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Skip step if gradients have collapsed
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                scaler.step(optimizer)
                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f'{loss.item():.4f}')
            else:
                pbar.set_postfix(loss='gradNaN (skip)')

            scaler.update()

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ── Validation ──
        model.eval()
        total_iou = 0.0
        total_dice = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='  Val', unit='batch', leave=False):
                imgs = batch['image'].to(
                    device, dtype=torch.float32, memory_format=torch.channels_last
                )
                masks = batch['mask'].to(device, dtype=torch.long)
                with torch.autocast(
                    device.type if device.type != 'mps' else 'cpu', enabled=amp
                ):
                    pred = model(imgs)
                pred_bin = pred.argmax(dim=1)
                iou, dice = binary_metrics(pred_bin, masks)
                total_iou += iou
                total_dice += dice

        avg_iou = total_iou / max(len(val_loader), 1)
        avg_dice = total_dice / max(len(val_loader), 1)
        scheduler.step(avg_dice)

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['val_iou'].append(avg_iou)
        history['val_dice'].append(avg_dice)

        logging.info(f'Epoch {epoch:3d} — Loss: {avg_loss:.4f}  IoU: {avg_iou:.4f}  Dice: {avg_dice:.4f}')

        # ── Early stopping (by Dice) ──
        if avg_dice > best_dice + early_stop_delta:
            best_dice = avg_dice
            best_state = copy.deepcopy(model.state_dict())
            if save_checkpoint:
                torch.save(model.state_dict(), str(DIR_CKPT / 'best_binary.pth'))
            stale_epochs = 0
            logging.info(f'  ★ New best — Dice: {best_dice:.4f}')
        else:
            stale_epochs += 1

        if stale_epochs >= early_stop_patience:
            logging.info(f'Early stopping triggered at epoch {epoch}')
            break

    # ── Save final ──
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_checkpoint:
        torch.save(
            {'model': best_state or model.state_dict(), 'history': history},
            str(DIR_CKPT / 'binary_unet_complete.pth'),
        )
    logging.info(f'Training complete. Best validation Dice: {best_dice:.4f}')
    return history


# ===========================================================================
# 5. CLI
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Train U-Net for binary leaf segmentation (2 classes)'
    )
    p.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    p.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    p.add_argument('--lr', '-l', type=float, default=1e-5, help='Learning rate')
    p.add_argument('--validation', '-v', type=float, default=10.0,
                   help='Validation percentage (0-100)')
    p.add_argument('--amp', action='store_true', default=True,
                   help='Use mixed precision (default: True)')
    p.add_argument('--no-amp', action='store_false', dest='amp',
                   help='Disable mixed precision')
    p.add_argument('--early-stop-patience', type=int, default=15,
                   help='Epochs without improvement before early stop')
    p.add_argument('--early-stop-delta', type=float, default=0.005,
                   help='Minimum Dice increase to reset patience')
    p.add_argument('--bilinear', action='store_true', default=False,
                   help='Use bilinear upsampling')
    return p.parse_args()


# ===========================================================================
# 6. Main
# ===========================================================================

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    logging.info(f'Binary segmentation — 2 classes (background + leaf)')

    # ── Build model ──
    model = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    model.to(device)

    logging.info(
        f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t2 output channels (binary bg/leaf)\n'
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    # ── Train ──
    try:
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.validation / 100,
            amp=args.amp,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            'CUDA OOM — enabling gradient checkpointing. '
            'Reduce batch size or disable AMP.'
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.validation / 100,
            amp=args.amp,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )
