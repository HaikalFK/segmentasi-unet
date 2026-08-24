import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Stage2Visualizer:
    def __init__(self):
        self.instance_cmap = self._build_instance_cmap()

    @staticmethod
    def _build_instance_cmap():
        colors = ["#000000"]
        np.random.seed(42)
        for _ in range(255):
            colors.append(
                "#{:02x}{:02x}{:02x}".format(
                    np.random.randint(50, 256),
                    np.random.randint(50, 256),
                    np.random.randint(50, 256),
                )
            )
        return ListedColormap(colors)

    def show_pipeline(self, result, figsize=(18, 6)):
        """Show 6-panel pipeline visualization."""
        titles = [
            "Input", "Probability", "Binary Mask",
            "Distance Map", "Markers", "Instance Mask",
        ]

        images = [
            result["image"],
            result["probability"],
            result["binary_mask"],
            result["distance_map"],
            result["markers"],
            result["instance_mask"],
        ]

        _, axes = plt.subplots(2, 3, figsize=figsize)
        for ax, img, title in zip(axes.flat, images, titles):
            ax.set_title(title)
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def show_overlay(self, result, figsize=(10, 5)):
        """Show instance mask overlay on original image."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)  # noqa: F841

        ax1.set_title("Original")
        ax1.imshow(result["image"])
        ax1.axis("off")

        ax2.set_title("Instance Overlay")
        ax2.imshow(result["image"])
        instance = result["instance_mask"]

        # Draw boundaries
        from skimage.segmentation import find_boundaries
        boundaries = find_boundaries(instance, mode="thick")
        overlay = result["image"].copy().astype(np.float32)
        overlay[boundaries] = [255, 0, 0]
        ax2.imshow(overlay.astype(np.uint8))

        # Label instances
        from scipy.ndimage import center_of_mass
        for inst_id in np.unique(instance):
            if inst_id == 0:
                continue
            cy, cx = center_of_mass(instance == inst_id)
            ax2.text(cx, cy, str(inst_id), color="white", fontsize=10,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        ax2.axis("off")
        plt.tight_layout()
        plt.show()
