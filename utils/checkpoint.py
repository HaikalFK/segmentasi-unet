"""Minimal checkpoint manager for segmentation experiments.

Design principle: Save only what's needed for reproduction & evaluation.
- model.state_dict() only (~18 MB for U-Net)
- config.yaml for full reproducibility
- metrics.json for learning curves
NO optimizer/scheduler states, NO per-epoch checkpoints.
"""

import torch
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MinimalCheckpoint:
    """
    Lightweight checkpoint manager.

    Per experiment (per loss function) saves only 3 files:
    - best_{loss_name}.pth     : model state_dict only (~18 MB)
    - config_{loss_name}.yaml  : full hyperparameters (~1 KB)
    - metrics_{loss_name}.json : history per epoch (~5 KB)
    """

    def __init__(self, save_dir: Path, experiment_name: str = "experiment"):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_best(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        metrics_history: Dict[str, list],
        loss_name: str,
        best_metric_value: float,
        best_metric_name: str = "val_dice",
    ):
        """
        Save best model + metadata.

        Args:
            model: trained model (state_dict extracted)
            config: full training config dict
            metrics_history: dict of lists, e.g., {"train_loss": [...], "val_dice": [...], "val_iou": [...]}
            loss_name: identifier for this loss function (e.g., "standard_bce")
            best_metric_value: the best validation metric achieved
            best_metric_name: name of the metric used for "best"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Model state_dict only (no optimizer, no scheduler, no epoch)
        model_path = self.save_dir / f"best_{loss_name}.pth"
        torch.save(
            model.state_dict(),
            model_path,
            _use_new_zipfile_serialization=True,  # compressed
        )

        # 2. Config for full reproducibility
        config_path = self.save_dir / f"config_{loss_name}.yaml"
        config_with_meta = {
            **config,
            "_meta": {
                "loss_name": loss_name,
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
                "saved_at": timestamp,
                "experiment": self.experiment_name,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, sort_keys=False)

        # 3. Metrics history for plotting
        metrics_path = self.save_dir / f"metrics_{loss_name}.json"
        metrics_data = {
            "history": metrics_history,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_epoch": metrics_history[best_metric_name].index(best_metric_value) + 1,
            "total_epochs": len(metrics_history.get("train_loss", [])),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        size_mb = model_path.stat().st_size / 1e6
        print(f"  ✓ Saved: {model_path.name} ({size_mb:.1f} MB)")
        print(f"  ✓ Saved: {config_path.name}")
        print(f"  ✓ Saved: {metrics_path.name}")

    @staticmethod
    def load_model(model: torch.nn.Module, path: Path, device: torch.device):
        """Load state_dict into existing model architecture."""
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        """Load config YAML."""
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_metrics(path: Path) -> Dict[str, Any]:
        """Load metrics JSON."""
        with open(path) as f:
            return json.load(f)

    def list_saved(self) -> Dict[str, Dict]:
        """List all saved checkpoints in this directory."""
        results = {}
        for model_path in self.save_dir.glob("best_*.pth"):
            loss_name = model_path.stem.replace("best_", "")
            config_path = self.save_dir / f"config_{loss_name}.yaml"
            metrics_path = self.save_dir / f"metrics_{loss_name}.json"

            entry = {"model": model_path, "size_mb": model_path.stat().st_size / 1e6}
            if config_path.exists():
                entry["config"] = self.load_config(config_path)
            if metrics_path.exists():
                entry["metrics"] = self.load_metrics(metrics_path)
            results[loss_name] = entry
        return results

    @staticmethod
    def cleanup_all_models(save_dir: Path):
        """Delete all .pth files (keep configs & metrics)."""
        save_dir = Path(save_dir)
        deleted = []
        for pth in save_dir.glob("*.pth"):
            pth.unlink()
            deleted.append(pth.name)
        if deleted:
            print(f"  🗑 Deleted models: {', '.join(deleted)}")
        return deleted


class ComparisonCheckpoint:
    """
    Aggregates results from multiple loss function experiments
    and generates comparison artifacts.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.comparison_dir = self.base_dir / "comparison"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

    def aggregate(self, loss_names: list) -> Dict[str, Any]:
        """Collect metrics from all loss experiments."""
        aggregated = {}
        for loss_name in loss_names:
            metrics_path = self.base_dir / loss_name / f"metrics_{loss_name}.json"
            config_path = self.base_dir / loss_name / f"config_{loss_name}.yaml"
            model_path = self.base_dir / loss_name / f"best_{loss_name}.pth"

            if metrics_path.exists():
                data = MinimalCheckpoint.load_metrics(metrics_path)
                aggregated[loss_name] = {
                    "metrics": data,
                    "config": MinimalCheckpoint.load_config(config_path) if config_path.exists() else None,
                    "model_path": model_path if model_path.exists() else None,
                }
        return aggregated

    def save_comparison_csv(self, aggregated: Dict, filename: str = "metrics_comparison.csv"):
        """Save comparison table as CSV."""
        import csv

        rows = []
        for loss_name, data in aggregated.items():
            m = data["metrics"]
            row = {
                "loss_function": loss_name,
                "best_epoch": m["best_epoch"],
                "total_epochs": m["total_epochs"],
                "best_val_dice": m["best_metric_value"],
                "final_train_loss": m["history"].get("train_loss", [None])[-1],
                "final_val_dice": m["history"].get("val_dice", [None])[-1],
                "final_val_iou": m["history"].get("val_iou", [None])[-1],
            }
            rows.append(row)

        csv_path = self.comparison_dir / filename
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✓ Saved comparison CSV: {csv_path}")
        return csv_path

    def save_comparison_md(self, aggregated: Dict, filename: str = "comparison_summary.md"):
        """Save comparison as Markdown table."""
        lines = [
            "# Loss Function Comparison Summary",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Final Metrics",
            "",
            "| Loss Function | Best Epoch | Best Val Dice | Final Train Loss | Final Val Dice | Final Val IoU |",
            "|---------------|------------|---------------|------------------|----------------|---------------|",
        ]

        for loss_name, data in aggregated.items():
            m = data["metrics"]
            lines.append(
                f"| {loss_name} | {m['best_epoch']} | "
                f"{m['best_metric_value']:.4f} | "
                f"{m['history'].get('train_loss', [None])[-1]:.4f} | "
                f"{m['history'].get('val_dice', [None])[-1]:.4f} | "
                f"{m['history'].get('val_iou', [None])[-1]:.4f} |"
            )

        lines.extend([
            "",
            "## Learning Curves",
            "",
            f"![Loss Curves](loss_curves.png)",
            f"![Dice Curves](dice_curves.png)",
            f"![IoU Curves](iou_curves.png)",
            "",
        ])

        md_path = self.comparison_dir / filename
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ Saved comparison MD: {md_path}")
        return md_path

    def plot_comparison(self, aggregated: Dict):
        """Generate comparison plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loss_names = list(aggregated.keys())
        colors = {"standard_bce": "#E74C3C", "weighted_bce_dice": "#2E86AB"}

        # Plot 1: Training Loss
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        ax = axes[0]
        for loss_name in loss_names:
            history = aggregated[loss_name]["metrics"]["history"]
            epochs = list(range(1, len(history["train_loss"]) + 1))
            ax.plot(epochs, history["train_loss"], "o-", label=loss_name, color=colors.get(loss_name), linewidth=2, markersize=5)
        ax.set_title("Training Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

        # Dice
        ax = axes[1]
        for loss_name in loss_names:
            history = aggregated[loss_name]["metrics"]["history"]
            epochs = list(range(1, len(history["val_dice"]) + 1))
            ax.plot(epochs, history["val_dice"], "s-", label=loss_name, color=colors.get(loss_name), linewidth=2, markersize=5)
        ax.set_title("Validation Dice", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Score")
        ax.grid(alpha=0.3)
        ax.legend()

        # IoU
        ax = axes[2]
        for loss_name in loss_names:
            history = aggregated[loss_name]["metrics"]["history"]
            epochs = list(range(1, len(history["val_iou"]) + 1))
            ax.plot(epochs, history["val_iou"], "^-", label=loss_name, color=colors.get(loss_name), linewidth=2, markersize=5)
        ax.set_title("Validation IoU", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.suptitle("Loss Function Comparison: Standard BCE vs Weighted BCE+Dice", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = self.comparison_dir / "loss_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved comparison plot: {plot_path}")

        # Individual curves
        for metric_key, title, fname in [
            ("train_loss", "Training Loss", "loss_curves.png"),
            ("val_dice", "Validation Dice", "dice_curves.png"),
            ("val_iou", "Validation IoU", "iou_curves.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for loss_name in loss_names:
                history = aggregated[loss_name]["metrics"]["history"]
                epochs = list(range(1, len(history[metric_key]) + 1))
                ax.plot(epochs, history[metric_key], "o-", label=loss_name, color=colors.get(loss_name), linewidth=2, markersize=6)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title.split()[-1])
            ax.grid(alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.comparison_dir / fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved: {fname}")

        return plot_path