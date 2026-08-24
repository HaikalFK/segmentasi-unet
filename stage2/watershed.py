import numpy as np
from skimage.segmentation import watershed as sk_watershed


class MarkerControlledWatershed:
    def __init__(self, compactness=0.0):
        self.compactness = compactness

    def segment(self, distance_map, markers, binary_mask):
        """Marker-controlled watershed segmentation.

        Args:
            distance_map: float [H, W] distance transform
            markers: int32 [H, W] labeled markers
            binary_mask: uint8 [H, W] binary mask

        Returns:
            int32 [H, W] instance mask (0 = background)
        """
        # Invert distance so watershed floods from peaks
        inverted = distance_map.max() - distance_map

        # Mask background to prevent watershed from expanding there
        inverted[binary_mask == 0] = 0

        instance_mask = sk_watershed(
            inverted,
            markers=markers,
            mask=binary_mask,
            compactness=self.compactness,
        )

        return instance_mask.astype(np.int32)
