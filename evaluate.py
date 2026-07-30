import argparse
import logging
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet import UNet
from torch.utils.data import DataLoader, random_split


def iou_score(pred, target, n_classes, ignore_index=0):
    """
    Compute IoU (Intersection over Union) for each class and mean IoU.

    Args:
        pred: Predicted masks (B, H, W) or one-hot (B, C, H, W)
        target: Ground truth masks (B, H, W) or one-hot (B, C, H, W)
        n_classes: Number of classes
        ignore_index: Class index to ignore (typically background=0)

    Returns:
        iou_per_class: IoU for each class (excluding ignored)
        mean_iou: Mean IoU across classes (excluding ignored)
    """
    # Convert to one-hot if needed
    if pred.dim() == 3:
        pred = F.one_hot(pred, n_classes).permute(0, 3, 1, 2).float()
    if target.dim() == 3:
        target = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()

    # Exclude background class if specified
    if ignore_index is not None:
        pred = torch.cat([pred[:, :ignore_index], pred[:, ignore_index+1:]], dim=1)
        target = torch.cat([target[:, :ignore_index], target[:, ignore_index+1:]], dim=1)
        n_classes = n_classes - 1

    # Compute intersection and union
    intersection = (pred * target).sum(dim=(0, 2, 3))  # (C,)
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection  # (C,)

    # Avoid division by zero
    iou = intersection / (union + 1e-8)

    mean_iou = iou.mean()

    return iou, mean_iou


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_iou_per_class = None
    total_mean_iou = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            n_classes_net = net.n_classes

            if n_classes_net == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < n_classes_net, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, n_classes_net).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes_net).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # compute IoU
                pred_masks = mask_pred.argmax(dim=1)  # (B, H, W)
                iou_per_class, mean_iou = iou_score(pred_masks, mask_true.argmax(dim=1), n_classes_net, ignore_index=0)

                if total_iou_per_class is None:
                    total_iou_per_class = iou_per_class.cpu()
                else:
                    total_iou_per_class += iou_per_class.cpu()
                total_mean_iou += mean_iou.item()

    net.train()

    avg_dice = dice_score / max(num_val_batches, 1)
    avg_iou_per_class = total_iou_per_class / max(num_val_batches, 1) if total_iou_per_class is not None else None
    avg_mean_iou = total_mean_iou / max(num_val_batches, 1) if total_mean_iou > 0 else 0

    return avg_dice, avg_iou_per_class, avg_mean_iou


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Load model from checkpoint
    if not args.load:
        logging.error('Please provide a checkpoint file with --load')
        sys.exit(1)

    state_dict = torch.load(args.load, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    n_classes_from_ckpt = len(mask_values)

    if n_classes_from_ckpt != args.classes:
        logging.info(f'Overriding --classes {args.classes} -> {n_classes_from_ckpt} based on checkpoint mask_values')
        n_classes = n_classes_from_ckpt
    else:
        n_classes = args.classes

    # Create model
    net = UNet(n_channels=3, n_classes=n_classes, bilinear=args.bilinear)
    net.to(device=device)
    net.load_state_dict(state_dict)

    logging.info(f'Model loaded from {args.load}')
    logging.info(f'Number of classes: {n_classes}')
    logging.info(f'Mask values: {mask_values}')

    # Create dataset
    try:
        dataset = CarvanaDataset('./data/imgs/', './data/masks/', args.scale, target_size=(256, 256))
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset('./data/imgs/', './data/masks/', args.scale, target_size=(256, 256))

    # Override mask_values if pre-computed
    dataset.mask_values = mask_values

    # Split into train / validation
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loader
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'Validation set size: {n_val}')

    # Evaluate
    dice, iou_per_class, mean_iou = evaluate(net, val_loader, device, args.amp, n_classes)

    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Dice Score (excl. background): {dice:.4f}')
    print(f'Mean IoU (excl. background):   {mean_iou:.4f}')
    print('-'*60)
    if iou_per_class is not None:
        print('IoU per class (excl. background):')
        for i, iou in enumerate(iou_per_class):
            class_name = f'Class {i+1}' if i+1 < len(mask_values) else f'Class {i+1}'
            if i+1 < len(mask_values):
                class_name = f'Class {mask_values[i+1]}'
            print(f'  {class_name}: {iou:.4f}')
    print('='*60)