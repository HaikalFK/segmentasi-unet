import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import label


class MarkerGenerator:
    def __init__(self, min_distance=10, threshold_rel=0.2):
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel

    def generate(self, distance_map, binary_mask):
        """Find local maxima in distance map as instance center markers.

        Args:
            distance_map: float [H, W] from DistanceTransformer
            binary_mask: uint8 [H, W] binary mask

        Returns:
            int32 [H, W] marker labels (0 = background/unlabeled)
        """
        if not binary_mask.any():
            return np.zeros_like(binary_mask, dtype=np.int32)

        coords = peak_local_max(
            distance_map,
            min_distance=self.min_distance,
            threshold_rel=self.threshold_rel,
            labels=binary_mask,
        )

        markers = np.zeros(distance_map.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        markers = label(markers)[0]
        return markers.astype(np.int32)
