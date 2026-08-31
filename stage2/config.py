from dataclasses import dataclass


@dataclass
class WatershedConfig:
    probability_threshold: float = 0.5
    min_distance: int = 10
    foreground_threshold: float = 0.5
    min_instance_area: int = 100
    compactness: float = 0.0
    target_size: tuple = (256, 256)
