import sys
import subprocess
import collections
import collections.abc
import contextlib
import io
from pathlib import Path
from copy import copy, deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.feature.dedode.dedode_models import get_descriptor

from d2net_all import _D2Net, process_multiscale

from hloc import extractors
from hloc.utils.base_model import dynamic_load


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


class DeDoDeV2DedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dedode_path = Path(
            self._get_cfg_param(
                "dedode_path",
                "/home/minhnxh/Documents/VinRobotic/DeDoDe",
            )
        )
        dedode_lib_path = str(dedode_path / "build" / "lib")
        if dedode_lib_path not in sys.path:
            sys.path.insert(0, dedode_lib_path)

        from DeDoDe import dedode_detector_L

        detector_weights_path = Path(
            self._get_cfg_param(
                "detector_weights_path",
                dedode_path / "workspace" / "dedode_detector_L_v2.pth",
            )
        )
        if not detector_weights_path.exists():
            raise FileNotFoundError(detector_weights_path)

        detector_weights = torch.load(detector_weights_path, map_location=device)
        self.detector = dedode_detector_L(
            device=device,
            weights=detector_weights,
        ).eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape
        max_n = n if n > 0 else self.max_keypoints

        with torch.no_grad():
            detector_images = self.detector.normalizer(images)
            detections = self.detector.detect(
                {"image": detector_images},
                num_keypoints=max_n,
            )

        keypoints_norm = detections["keypoints"]
        confidence = detections["confidence"]
        keypoints = self.detector.to_pixel_coords(keypoints_norm, H, W)[0].to(
            device=device,
            dtype=torch.float32,
        )
        scores = confidence[0].to(device=device, dtype=torch.float32)

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            sampled = F.grid_sample(
                descriptions.float(),
                keypoints_norm[:, None].to(device=device, dtype=torch.float32),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class SuperPointDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = "SuperPoint_dedode"
        sp_conf = {
            "name": "superpoint",
            "nms_radius": self._get_cfg_param("nms_radius", 3),
            "max_keypoints": self._get_cfg_param("max_keypoints",
                                                 4096),
        }
        Model = dynamic_load(extractors, "superpoint")
        self.detector = Model(sp_conf).eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type",
                                              "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
        descriptor_url = "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth"
        if descriptor_type == "G":
            descriptor_url = "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"

        self.descriptor.load_state_dict(
            torch.hub.load_state_dict_from_url(descriptor_url,map_location="cpu")
        )

        self.eval()


    def forward(self, data):
        return self.keypoint_features(data)


    def _get_cfg_param(self, key, default):
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)


    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        gray = images.mean(dim=1, keepdim=True)

        with torch.no_grad():
            sp_pred = self.detector({"image": gray})

        keypoints = sp_pred["keypoints"][0].to(device,
                                               dtype=torch.float32)
        scores = sp_pred.get("scores", torch.ones(1,
                                                  keypoints.shape[0], device=device))[0]

        if n > 0 and keypoints.shape[0] > n:
            idx = torch.randperm(keypoints.shape[0],
                                 generator=generator, device=device)[:n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)

        with torch.no_grad():
            descriptions = self.descriptor.forward(images)

            wh = torch.tensor([W, H], device=device,
                              dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            kp_grid = kp_norm.unsqueeze(1)

            sampled = F.grid_sample(
                descriptions.float(),
                kp_grid,
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class R2D2DedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)

        r2d2_conf = {
            "name": "r2d2",
            "max_keypoints": self.max_keypoints,
        }
        Model = dynamic_load(extractors, "r2d2")
        self.detector = Model(r2d2_conf).eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        with torch.no_grad():
            pred = self.detector({"image": images})

        keypoints = pred["keypoints"][0].to(device, dtype=torch.float32)
        scores = pred.get("scores", torch.ones(1, keypoints.shape[0], device=device))[0]

        max_n = n if n > 0 else self.max_keypoints
        if max_n > 0 and keypoints.shape[0] > max_n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:max_n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class AcceleratedFeaturesDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        self.detection_threshold = self._get_cfg_param("detection_threshold", 0.05)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        accelerated_features_path = Path(
            self._get_cfg_param(
                "accelerated_features_path",
                "/home/minhnxh/Documents/VinRobotic/accelerated_features",
            )
        )
        weights_path = Path(
            self._get_cfg_param(
                "weights_path",
                accelerated_features_path / "weights" / "xfeat.pt",
            )
        )
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)

        af_path = str(accelerated_features_path)
        if af_path not in sys.path:
            sys.path.insert(0, af_path)

        from modules.xfeat import XFeat

        self.detector = XFeat(
            weights=str(weights_path),
            top_k=self.max_keypoints,
            detection_threshold=self.detection_threshold,
        )
        self.detector = self.detector.eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape
        max_n = n if n > 0 else self.max_keypoints

        with torch.no_grad():
            pred = self.detector.detectAndCompute(
                images,
                top_k=max_n,
                detection_threshold=self.detection_threshold,
            )[0]

        keypoints = pred["keypoints"].to(device=device, dtype=torch.float32)
        scores = pred["scores"].to(device=device, dtype=torch.float32)

        if keypoints.shape[0] == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class DiskDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)

        disk_conf = {
            "name": "disk",
            "max_keypoints": self.max_keypoints,
        }
        Model = dynamic_load(extractors, "disk")
        self.detector = Model(disk_conf).eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        with torch.no_grad():
            pred = self.detector({"image": images})

        keypoints = pred["keypoints"][0].to(device, dtype=torch.float32)
        scores = pred.get("scores", torch.ones(1, keypoints.shape[0], device=device))[0]

        max_n = n if n > 0 else self.max_keypoints
        if max_n > 0 and keypoints.shape[0] > max_n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:max_n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class SIFTDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        self.detector_type = self._get_cfg_param("detector_type", "sift").lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.detector_type == "sift" and hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create(nfeatures=self.max_keypoints)
        elif self.detector_type in {"orb", "sift"}:
            self.detector_type = "orb"
            self.detector = cv2.ORB_create(nfeatures=self.max_keypoints)
        else:
            raise ValueError(f"Unsupported OpenCV detector: {self.detector_type}")

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        gray = images[0].mean(dim=0).detach().cpu().numpy()
        gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        cv_keypoints = self.detector.detect(gray, None)

        if len(cv_keypoints) == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)
        else:
            cv_keypoints = sorted(cv_keypoints, key=lambda kp: kp.response, reverse=True)
            max_n = n if n > 0 else self.max_keypoints
            if max_n > 0:
                cv_keypoints = cv_keypoints[:max_n]

            keypoints_np = np.array([kp.pt for kp in cv_keypoints], dtype=np.float32)
            scores_np = np.array([kp.response for kp in cv_keypoints], dtype=np.float32)
            keypoints = torch.from_numpy(keypoints_np).to(device=device, dtype=torch.float32)
            scores = torch.from_numpy(scores_np).to(device=device, dtype=torch.float32)

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class GoodFeaturesToTrackDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        self.quality_level = self._get_cfg_param("quality_level", 0.01)
        self.min_distance = self._get_cfg_param("min_distance", 3)
        self.block_size = self._get_cfg_param("block_size", 3)
        self.use_harris = self._get_cfg_param("use_harris", False)
        self.k = self._get_cfg_param("k", 0.04)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        gray = images[0].mean(dim=0).detach().cpu().numpy()
        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        max_n = n if n > 0 else self.max_keypoints
        corners = cv2.goodFeaturesToTrack(
            gray_u8,
            maxCorners=max_n,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            useHarrisDetector=self.use_harris,
            k=self.k,
        )

        if corners is None or len(corners) == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)
        else:
            keypoints_np = corners.reshape(-1, 2).astype(np.float32)
            response_map = cv2.cornerHarris(gray_u8, self.block_size, 3, self.k)
            xs = np.clip(np.round(keypoints_np[:, 0]).astype(np.int64), 0, W - 1)
            ys = np.clip(np.round(keypoints_np[:, 1]).astype(np.int64), 0, H - 1)
            scores_np = response_map[ys, xs].astype(np.float32)
            keypoints = torch.from_numpy(keypoints_np).to(device=device, dtype=torch.float32)
            scores = torch.from_numpy(scores_np).to(device=device, dtype=torch.float32)

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class SiLKDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.silk_path = Path(
            self._get_cfg_param(
                "silk_path",
                "/home/minhnxh/Documents/VinRobotic/silk",
            )
        )
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        self.threshold = self._get_cfg_param("threshold", 1.0)
        self.nms_dist = self._get_cfg_param("nms_dist", 0)
        self.border_dist = self._get_cfg_param("border_dist", 0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = Path(
            self._get_cfg_param(
                "checkpoint",
                self.silk_path / "assets" / "models" / "pvgg-4.ckpt",
            )
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SiLK checkpoint not found at {checkpoint}. "
                "Run the SiLK asset download script or pass a valid checkpoint path."
            )

        silk_lib_path = str(self.silk_path / "lib")
        if silk_lib_path not in sys.path:
            sys.path.insert(0, silk_lib_path)

        from silk.backbones.silk.silk import SiLKVGG, from_feature_coords_to_image_coords
        from silk.backbones.superpoint.vgg import ParametricVGG
        from silk.config.model import load_model_from_checkpoint

        self.from_feature_coords_to_image_coords = from_feature_coords_to_image_coords
        backbone = ParametricVGG(
            input_num_channels=1,
            use_max_pooling=False,
            padding=0,
            channels=(64, 64, 128, 128),
            normalization_fn=[
                torch.nn.BatchNorm2d(64),
                torch.nn.BatchNorm2d(64),
                torch.nn.BatchNorm2d(128),
                torch.nn.BatchNorm2d(128),
            ],
        )
        detector = SiLKVGG(
            in_channels=1,
            detection_threshold=self.threshold,
            detection_top_k=self.max_keypoints,
            nms_dist=self.nms_dist,
            padding=0,
            border_dist=self.border_dist,
            descriptor_scale_factor=1.41,
            default_outputs=("sparse_positions", "sparse_descriptors"),
            backbone=deepcopy(backbone),
        )
        self.detector = load_model_from_checkpoint(
            detector,
            checkpoint_path=str(checkpoint),
            state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
            device=device,
            freeze=True,
            eval=True,
        )

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape
        gray = images.mean(dim=1, keepdim=True)

        with torch.no_grad():
            positions, _ = self.detector(gray)
            positions = self.from_feature_coords_to_image_coords(self.detector, positions)[0]

        if positions.numel() == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)
        else:
            keypoints = positions[:, [1, 0]].to(device=device, dtype=torch.float32)
            scores = positions[:, 2].to(device=device, dtype=torch.float32)

        max_n = n if n > 0 else self.max_keypoints
        if max_n > 0 and keypoints.shape[0] > max_n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:max_n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class AffNetDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.affnet_path = Path(
            self._get_cfg_param(
                "affnet_path",
                "/home/minhnxh/Documents/VinRobotic/affnet",
            )
        )

        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # affnet_path = str(self.affnet_path)
        # if affnet_path not in sys.path:
        #     sys.path.append(str(affnet_path))
        #
        # from architectures import AffNetFast
        # from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor

        sys.path.insert(0, str(self.affnet_path))
        try:
            from architectures import AffNetFast
            from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
        finally:
            try:
                sys.path.remove(str(self.affnet_path))
            except ValueError:
                pass

        affnet = AffNetFast(PS=32)
        weights = Path(
            self._get_cfg_param(
                "affnet_weights",
                self.affnet_path / "pretrained" / "AffNet.pth",
            )
        )
        checkpoint = torch.load(weights, map_location=device)
        affnet.load_state_dict(checkpoint["state_dict"])
        affnet.eval()

        self.detector = ScaleSpaceAffinePatchExtractor(
            mrSize=5.192,
            num_features=self.max_keypoints,
            border=5,
            num_Baum_iters=1,
            th=self._get_cfg_param("threshold", None),
            AffNet=affnet,
        ).eval().to(device)

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        gray = images.mean(dim=1, keepdim=True) * 255.0
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            lafs, scores = self.detector(gray)

        if lafs.numel() == 0:
            keypoints = torch.rand((100, 2), device=device) * torch.tensor([W, H], device=device)
            scores = torch.ones((100,), device=device, dtype=torch.float32)
        else:
            keypoints = lafs[:, :, 2].to(device, dtype=torch.float32)
            scores = scores.to(device, dtype=torch.float32)

        max_n = n if n > 0 else self.max_keypoints
        if max_n > 0 and keypoints.shape[0] > max_n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:max_n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
        }


class ASLFeatDedodeEncoder(nn.Module):
    out_channels = 256
    sparse = True

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}
        self.aslfeat_path = Path(
            self._get_cfg_param(
                "aslfeat_path",
                "/home/minhnxh/Documents/VinRobotic/ASLFeat",
            )
        )
        self.max_keypoints = self._get_cfg_param("max_keypoints", 5000)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hasattr(collections, "Mapping"):
            collections.Mapping = collections.abc.Mapping

        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        if not hasattr(tf, "placeholder"):
            tf.placeholder = tf.compat.v1.placeholder

        sys.path.insert(0, str(self.aslfeat_path))
        try:
            from models import get_model
        finally:
            try:
                sys.path.remove(str(self.aslfeat_path))
            except ValueError:
                pass

        model_path = Path(
            self._get_cfg_param(
                "aslfeat_model_path",
                self.aslfeat_path / "pretrained" / "aslfeat" / "model.ckpt-380000",
            )
        )
        if not (model_path.with_suffix(model_path.suffix + ".index").exists() or Path(str(model_path) + ".index").exists()):
            raise FileNotFoundError(
                f"ASLFeat checkpoint not found at {model_path}. "
                "Download/copy the ASLFeat pretrained checkpoint there or pass aslfeat_model_path."
            )

        self.detector = get_model("feat_model")(
            str(model_path),
            max_dim=self._get_cfg_param("max_dim", 2048),
            config={
                "kpt_n": self.max_keypoints,
                "kpt_refinement": True,
                "deform_desc": 1,
                "score_thld": self._get_cfg_param("score_thld", 0.5),
                "edge_thld": self._get_cfg_param("edge_thld", 10),
                "multi_scale": self._get_cfg_param("multi_scale", False),
                "multi_level": True,
                "nms_size": self._get_cfg_param("nms_size", 3),
                "eof_mask": 5,
                "need_norm": True,
                "use_peakiness": True,
            },
        )

        descriptor_type = self._get_cfg_param("descriptor_type", "B")
        self.descriptor = get_descriptor(descriptor_type).to(device)
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
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def keypoint_features(self, data, n=0, generator=None):
        assert not self.training

        images = data["image"]
        device = images.device
        _, _, H, W = images.shape

        gray = images[0].mean(dim=0).detach().cpu().numpy()
        gray = (gray * 255.0).astype(np.float32)[..., np.newaxis]
        _, kpts_np, scores_np = self.detector.run_test_data(gray)

        keypoints = torch.from_numpy(kpts_np).to(device=device, dtype=torch.float32)
        scores = torch.from_numpy(scores_np).to(device=device, dtype=torch.float32)

        max_n = n if n > 0 else self.max_keypoints
        if max_n > 0 and keypoints.shape[0] > max_n:
            idx = torch.randperm(keypoints.shape[0], generator=generator, device=device)[:max_n]
            keypoints = keypoints[idx]
            scores = scores[idx]

        keypoints_batched = keypoints.unsqueeze(0)
        with torch.no_grad():
            descriptions = self.descriptor.forward(images)
            wh = torch.tensor([W, H], device=device, dtype=keypoints_batched.dtype)
            kp_norm = (keypoints_batched / wh) * 2 - 1
            sampled = F.grid_sample(
                descriptions.float(),
                kp_norm.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )

        return {
            "keypoints": keypoints_batched,
            "scores": scores.unsqueeze(0),
            "descriptors": sampled[:, :, 0, :].detach(),
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
