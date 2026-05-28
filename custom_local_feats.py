import sys
import subprocess
from pathlib import Path
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature.dedode.dedode_models import get_descriptor

from d2net_all import _D2Net, process_multiscale


# Inherit from nn.Module to natively get .eval(), .training, and proper device routing
class D2NetDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True
    d2net_path = Path("third_party/d2net")

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        # Fallback dummy configuration dict to avoid crashes if empty
        self.cfg = cfg if cfg is not None else {}

        # 1. Download and Initialize D2-Net Detector Components
        model_name = self._get_cfg_param("d2_model_name", "d2_tf.pth")
        checkpoint_dir = Path(self._get_cfg_param("d2_checkpoint_dir", self.d2net_path / "models"))
        model_file = checkpoint_dir / model_name

        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True, parents=True)
            cmd = [
                "wget",
                "https://dusmanu.com/files/d2-net/" + model_name,
                "-O",
                str(model_file),
            ]
            subprocess.run(cmd, check=True)

        self.d2_net = _D2Net(
            model_file=model_file,
            use_relu=self._get_cfg_param("d2_use_relu", True),
            use_cuda=torch.cuda.is_available()
        )

        # 2. Initialize DeDoDe Descriptor Network
        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to("cuda" if torch.cuda.is_available() else "cpu")

        descriptor_url = "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth"
        if descriptor_type == "G":
            descriptor_url = "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"

        self.descriptor.load_state_dict(
            torch.hub.load_state_dict_from_url(descriptor_url, map_location="cpu")
        )

        self.eval()

    def forward(self, data):
        return self.keypoint_features(data)

    def _get_cfg_param(self, key, default):
        """Helper to safely fetch configurations from object or dict structures."""
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        """
        Extracts keypoints using D2-Net and samples descriptors using DeDoDe.
        Expected data['image']: Tensor [B, 3, H, W], normalized to [0, 1] range.
        """
        assert not self.training

        images = data["image"]  # [B, 3, H, W]
        device = images.device
        B, C, H, W = images.shape

        # --- PART 1: D2-NET KEYPOINT DETECTOR ---
        d2_image = images.flip(1)  # RGB -> BGR
        # FIX: Ensure norm tensor matches the target device
        norm = d2_image.new_tensor([103.939, 116.779, 123.68], device=device)
        d2_image = d2_image * 255 - norm.view(1, 3, 1, 1)

        multiscale = self._get_cfg_param("d2_multiscale", False)
        scales = [.5, 1, 2] if multiscale else [1]

        kp_np, scores_np, _ = process_multiscale(d2_image, self.d2_net, scales=scales)

        keypoints = torch.from_numpy(kp_np[:, [1, 0]]).to(device, dtype=torch.float32)  # [N, 2]
        scores = torch.from_numpy(scores_np).to(device, dtype=torch.float32)

        if keypoints.shape[0] == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)

        if n > 0 and keypoints.shape[0] > n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:n]
            keypoints = keypoints[idx, :]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)

        # --- PART 2: DEDODE DESCRIPTOR SAMPLING ---
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)  # [B, D, H_feat, W_feat]

        wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
        kp_norm = (keypoints_batched / wh) * 2 - 1
        kp_grid = kp_norm.unsqueeze(1)  # Shape: [1, 1, N, 2]

        with torch.no_grad():
            sampled_descriptions = F.grid_sample(
                descriptions.float(),
                kp_grid,
                mode="bilinear",
                align_corners=False,
            )  # Output: [1, D, 1, N]

        final_descriptors = sampled_descriptions[:, :, 0, :].detach()  # [1, 256, N]

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": final_descriptors,
        }


if __name__ == '__main__':
    # Initialize the architecture with dummy config object/dict
    config = {"descriptor_type": "B", "d2_multiscale": False}
    m = D2NetDedodeEncoder(cfg=config)

    # Send model to target processing unit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)

    # Simulate an input batch tensor (1 image, 3 channels, 224x224 px)
    mock_data = {
        "image": torch.rand((1, 3, 224, 224), device=device)
    }
    with torch.no_grad():
        outputs = m.keypoint_features(mock_data)
    print("Keypoints shape:", outputs["keypoints"].shape)
    print("Descriptors shape:", outputs["descriptors"].shape)
