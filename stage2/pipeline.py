import os
import numpy as np
from scipy.ndimage import label as cc_label

from stage2.preprocessing import Stage2Preprocessor
from stage2.distance_transform import DistanceTransformer
from stage2.markers import MarkerGenerator
from stage2.watershed import MarkerControlledWatershed
from stage2.config import WatershedConfig


class Stage2Pipeline:
    def __init__(self, predictor, config=None):
        self.predictor = predictor
        self.config = config or WatershedConfig()
        self.preprocessor = Stage2Preprocessor(target_size=self.config.target_size)
        self.distance_transformer = DistanceTransformer()
        self.marker_generator = MarkerGenerator(
            min_distance=self.config.min_distance,
        )
        self.watershed = MarkerControlledWatershed(
            compactness=self.config.compactness,
        )

    def process(self, image):
        """Run full Stage 2 pipeline on a single image.

        Args:
            image: PIL Image or numpy array (H, W, 3)

        Returns:
            dict with keys: image, probability, binary_mask,
                            distance_map, markers, instance_mask
        """
        probability = self.predictor.predict_probability(image)

        binary_mask = (
            probability > self.config.probability_threshold
        ).astype(np.uint8)

        labeled, n_components = cc_label(binary_mask)
        for i in range(1, n_components + 1):
            if (labeled == i).sum() < self.config.min_instance_area:
                binary_mask[labeled == i] = 0

        distance_map = self.distance_transformer.transform(binary_mask)

        markers = self.marker_generator.generate(distance_map, binary_mask)

        instance_mask = self.watershed.segment(
            distance_map, markers, binary_mask,
        )

        return {
            "image": np.asarray(image),
            "probability": probability,
            "binary_mask": binary_mask,
            "distance_map": distance_map,
            "markers": markers,
            "instance_mask": instance_mask,
        }

    def process_batch(self, images):
        """Run pipeline on a list of images."""
        return [self.process(img) for img in images]

    def save_results(self, result, output_dir, prefix=""):
        """Save each stage output as numpy arrays."""
        os.makedirs(output_dir, exist_ok=True)

        save_map = {
            "probability.npy": result["probability"],
            "binary_mask.npy": result["binary_mask"],
            "distance_map.npy": result["distance_map"],
            "markers.npy": result["markers"],
            "instance_mask.npy": result["instance_mask"],
        }

        for name, data in save_map.items():
            path = os.path.join(output_dir, f"{prefix}{name}")
            np.save(path, data)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config=None, device=None,
                        n_channels=3, n_classes=2, bilinear=False):
        from stage2.prediction import Stage1Predictor
        target_size = config.target_size if config else (256, 256)
        predictor = Stage1Predictor.from_checkpoint(
            checkpoint_path,
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            device=device,
            target_size=target_size,
        )
        return cls(predictor, config)
