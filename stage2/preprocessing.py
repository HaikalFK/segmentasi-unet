import numpy as np
from PIL import Image


class Stage2Preprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def preprocess_image(self, image):
        """Preprocess RGB image for Stage 1 inference.

        Replicates BinaryLeafDataset._preprocess_img pipeline:
        BICUBIC resize -> numpy CHW float32 -> /255.0
        """
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")

        w, h = image.size
        new_w, new_h = self.target_size
        image = image.resize((new_w, new_h), Image.BICUBIC)

        img_np = np.asarray(image, dtype=np.float32)
        img_np = img_np.transpose(2, 0, 1)
        if img_np.max() > 1.0:
            img_np /= 255.0
        return img_np

    def preprocess_mask(self, mask, target_size=None):
        """Load instance mask without binarization (for evaluation).

        Returns original pixel values so instance identities are preserved.
        Pass target_size to match a prediction's (W, H); defaults to
        self.target_size.
        """
        if isinstance(mask, Image.Image):
            mask = mask.convert("L")
        else:
            mask = Image.fromarray(mask).convert("L")

        w, h = mask.size
        new_w, new_h = target_size if target_size else self.target_size
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        return np.asarray(mask, dtype=np.int64)
