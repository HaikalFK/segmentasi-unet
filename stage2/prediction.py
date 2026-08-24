import numpy as np
import torch
import torch.nn.functional as F

from unet.unet_model import UNet
from stage2.preprocessing import Stage2Preprocessor


class Stage1Predictor:
    def __init__(self, model, checkpoint_path, device, target_size=(256, 256)):
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.preprocessor = Stage2Preprocessor(target_size=target_size)
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)

    def predict_probability(self, image):
        """Returns probability map float32 [H, W] in [0, 1]."""
        img_tensor = self.preprocessor.preprocess_image(image)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(img_tensor)
            output = F.interpolate(
                output,
                size=(image.size[1], image.size[0]) if hasattr(image, "size")
                else (image.shape[0], image.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        return prob.astype(np.float32)

    def predict_binary(self, image, threshold=0.5):
        """Returns binary mask uint8 [H, W] with values {0, 1}."""
        prob = self.predict_probability(image)
        return (prob > threshold).astype(np.uint8)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, n_channels=3, n_classes=2,
                        bilinear=False, device=None, target_size=(256, 256)):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
        return cls(model, checkpoint_path, device, target_size)
