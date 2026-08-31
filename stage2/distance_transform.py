import numpy as np
from scipy.ndimage import distance_transform_edt


class DistanceTransformer:
    def transform(self, binary_mask):
        """Compute distance from each foreground pixel to nearest background.

        Args:
            binary_mask: uint8 [H, W] with values {0, 1}

        Returns:
            float64 [H, W] distance map, 0 at background pixels
        """
        return distance_transform_edt(binary_mask)
