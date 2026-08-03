"""Thin W&B integration for experiment tracking.

Graceful degradation: if wandb is missing OR config disables it, all calls
become no-ops, so experiment code never needs conditionals for W&B.

Usage:
    tracker = WandBTracker(cfg.get('wandb', {}), loss_name='standard_bce')
    tracker.init()
    tracker.log({'epoch': e, 'loss': l})
    tracker.log_image('prediction', img_tensor, caption='...')
    tracker.finish()

Design notes:
- One W&B *run* per loss function => 2 runs per experiment => compared
  side-by-side via `group`.
- Offline by default (matches project convention in train.py) — sync to
  cloud later with `wandb sync`.
- Model weights stay on disk (best_*.pth); W&B tracks metrics/curves/images.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _WANDB_AVAILABLE = False


class WandBTracker:
    def __init__(self, config: dict, loss_name: str):
        """config = the `wandb:` section of YAML; loss_name = run tags/names."""
        self.config = config or {}
        self.loss_name = loss_name
        self.enabled = bool(self.config.get("enabled", True)) and _WANDB_AVAILABLE
        self.mode = self.config.get("mode", "offline")
        self.run = None
        # Load-once module-level import stays inside the class path for clarity.

    @property
    def active(self) -> bool:
        """True if a wandb run is currently open and logging."""
        return self.enabled and self.run is not None and self.run is not False

    def init(self, extra_config: dict = None):
        """Start a wandb run for this loss function."""
        if not self.enabled:
            logger.info("[wandb] disabled — skipping run init for %s", self.loss_name)
            return None
        try:
            if self.mode == "disabled":
                # Do not even create the run; metrics-only local path.
                return None
            self.run = wandb.init(
                project=self.config.get("project", "unet-loss-comparison"),
                entity=self.config.get("entity") or None,
                group=self.config.get("group") or None,
                name=f"{self.loss_name}",
                job_type=self.loss_name,
                mode=self.mode,
                config=extra_config or {},
                # finish_previous=True avoids the deprecated `reinit` warning in wandb 0.25+
                finish_previous=True,
            )
            logger.info("[wandb] run %s started (mode=%s)", self.loss_name, self.mode)
            return self.run
        except Exception as e:
            logger.warning("[wandb] init failed (%s) — continuing without tracking", e)
            self.run = False
            return None

    def log(self, metrics: dict, step: int = None):
        """Log metrics to current wandb run. No-op if inactive."""
        if not self.active:
            return
        try:
            if step is not None:
                metrics = {**metrics, "step": step}
            self.run.log(metrics)
        except Exception as e:
            logger.warning("[wandb] log failed: %s", e)

    def finish(self):
        if self.active:
            try:
                self.run.finish()
            except Exception as e:
                logger.warning("[wandb] finish failed: %s", e)
        self.run = None

    @staticmethod
    def image(tensor):
        """Wrap a CHW float tensor as a wandb.Image (or passthrough if no wandb)."""
        if not _WANDB_AVAILABLE:
            return tensor
        return wandb.Image(tensor)