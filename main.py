import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import faiss
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from ace_loss import ReproLoss
from ace_network import Regressor, Head
from ace_util import read_and_preprocess
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from pykdtree.kdtree import KDTree
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from tqdm import tqdm

import colmap_read
from ace_util import (
    get_pixel_grid,
    to_homogeneous,
    read_nvm_file,
    localize_pose_lib,
)
from ace_util import normalize_shape
from ace_util import set_seed, _strtobool
from dataset import CamLocDataset

_logger = logging.getLogger(__name__)
sys.path.append("../MixVPR")
from mix_vpr_main import VPRModel
from mix_vpr_demo import load_image as load_image_mix_vpr


def read_args():
    parser = argparse.ArgumentParser(
        description="Fast training of a scene coordinate regression network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "scene",
        type=Path,
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "output_map_file", type=Path, help="target file for the trained network"
    )

    parser.add_argument(
        "--constraint_mask",
        type=int,
        default=1,
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--sampling_radius",
        type=int,
        default=5,
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--use_mask",
        type=bool,
        default=True,
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=False,
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--encoder_path",
        type=Path,
        default="../ace/ace_encoder_pretrained.pt",
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--num_head_blocks",
        type=int,
        default=1,
        help="depth of the regression head, defines the map size",
    )

    parser.add_argument(
        "--learning_rate_min",
        type=float,
        default=0.0005,
        help="lowest learning rate of 1 cycle scheduler",
    )

    parser.add_argument(
        "--learning_rate_max",
        type=float,
        default=0.005,
        help="highest learning rate of 1 cycle scheduler",
    )

    parser.add_argument(
        "--training_buffer_size",
        type=int,
        default=8000000,
        help="number of patches in the training buffer",
    )

    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=1024,
        help="number of patches drawn from each image when creating the buffer",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=5120,
        help="number of patches for each parameter update (has to be a multiple of 512)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="number of runs through the training buffer",
    )

    parser.add_argument(
        "--repro_loss_hard_clamp",
        type=int,
        default=1000,
        help="hard clamping threshold for the reprojection losses",
    )

    parser.add_argument(
        "--repro_loss_soft_clamp",
        type=int,
        default=50,
        help="soft clamping threshold for the reprojection losses",
    )

    parser.add_argument(
        "--repro_loss_soft_clamp_min",
        type=int,
        default=1,
        help="minimum value of the soft clamping threshold when using a schedule",
    )

    parser.add_argument(
        "--use_half", type=_strtobool, default=True, help="train with half precision"
    )

    parser.add_argument(
        "--use_homogeneous",
        type=_strtobool,
        default=True,
        help="train with half precision",
    )

    parser.add_argument(
        "--use_aug", type=_strtobool, default=True, help="Use any augmentation."
    )

    parser.add_argument(
        "--aug_rotation", type=int, default=15, help="max inplane rotation angle"
    )

    parser.add_argument("--aug_scale", type=float, default=1.5, help="max scale factor")

    parser.add_argument(
        "--image_resolution", type=int, default=480, help="base image resolution"
    )

    parser.add_argument(
        "--repro_loss_type",
        type=str,
        default="dyntanh",
        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
        help="Loss function on the reprojection error. Dyn varies the soft clamping threshold",
    )

    parser.add_argument(
        "--repro_loss_schedule",
        type=str,
        default="circle",
        choices=["circle", "linear"],
        help="How to decrease the softclamp threshold during training, circle is slower first",
    )

    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.1,
        help="enforce minimum depth of network predictions",
    )

    parser.add_argument(
        "--depth_target",
        type=float,
        default=10,
        help="default depth to regularize training",
    )

    parser.add_argument(
        "--depth_max",
        type=float,
        default=1000,
        help="enforce maximum depth of network predictions",
    )

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=None,
        help="split the training sequence in this number of clusters. disabled by default",
    )

    parser.add_argument(
        "--cluster_idx",
        type=int,
        default=None,
        help="train on images part of this cluster. required only if --num_clusters is set.",
    )

    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument(
        "--render_visualization",
        type=_strtobool,
        default=False,
        help="create a video of the mapping process",
    )

    parser.add_argument(
        "--render_target_path",
        type=Path,
        default="renderings",
        help="target folder for renderings, visualizer will create a subfolder with the map name",
    )

    parser.add_argument(
        "--render_flipped_portrait",
        type=_strtobool,
        default=False,
        help="flag for wayspots dataset where images are sideways portrait",
    )

    parser.add_argument(
        "--render_map_error_threshold",
        type=int,
        default=10,
        help="reprojection error threshold for the visualisation in px",
    )

    parser.add_argument(
        "--render_map_depth_filter",
        type=int,
        default=10,
        help="to clean up the ACE point cloud remove points too far away",
    )

    parser.add_argument(
        "--render_camera_z_offset",
        type=int,
        default=4,
        help="zoom out of the scene by moving render camera backwards, in meters",
    )
    return parser


def compute_pose(uv_arr, xyz_pred, focal_length, ppX, ppY, gt_pose_B44):
    pairs = []

    for j, (x, y) in enumerate(uv_arr):
        xy = [x, y]
        xyz = xyz_pred[j]
        pairs.append((xy, xyz))
    pose, info = localize_pose_lib(pairs, focal_length, ppX, ppY)

    est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
    est_pose = np.linalg.inv(est_pose)
    out_pose = torch.from_numpy(est_pose)

    # Calculate translation error.
    gt_pose_44 = gt_pose_B44[0]
    t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

    gt_R = gt_pose_44[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()

    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi

    return t_err, r_err


def predict_xyz(model, features_ori):
    feature_dim = features_ori.shape[-1]
    if type(features_ori) == np.ndarray:
        features_ori = torch.from_numpy(features_ori).cuda()
    amount = (
        features_ori.shape[0] // feature_dim + 1
    ) * feature_dim - features_ori.shape[0]
    fake_desc = torch.vstack(
        [features_ori, torch.zeros((amount, feature_dim), device="cuda")]
    )
    fake_desc = (
        fake_desc[None, None, ...].view(-1, 8, 16, feature_dim).permute(0, 3, 1, 2)
    )
    with torch.no_grad():
        pred_scene_coords_b3HW = model(fake_desc)
        pred_scene_coords_b3 = (
            pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).float()
        )
        xyz_pred = pred_scene_coords_b3[: features_ori.shape[0]].float().cpu().numpy()
    return xyz_pred


class TrainerACE:
    def __init__(self, options):
        self.mse_errors = None
        self.options = options

        self.device = torch.device("cuda")

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.ds_name = str(self.options.scene).split("/")[-1]
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.sfm_model_dir = None
        if "7scenes" in str(self.options.scene):
            self.ds_type = "7scenes"
            ds_name = str(self.options.scene).split("/")[-1].split("_")[-1]
            self.sfm_model_dir = (
                f"/home/n11373598/work/7scenes_reference_models/{ds_name}/sfm_gt"
            )
            _logger.info(f"Reading SFM from {self.sfm_model_dir}")
            self.recon_images = colmap_read.read_images_binary(
                f"{self.sfm_model_dir}/images.bin"
            )
            self.recon_cameras = colmap_read.read_cameras_binary(
                f"{self.sfm_model_dir}/cameras.bin"
            )
            self.recon_points = colmap_read.read_points3D_binary(
                f"{self.sfm_model_dir}/points3D.bin"
            )
            self.image_name2id = {}
            for image_id, image in self.recon_images.items():
                self.image_name2id[image.name.replace("/", "-")] = image_id
            self.image_id2points = {}
            for img_id in self.recon_images:
                pid_arr = self.recon_images[img_id].point3D_ids
                pid_arr = pid_arr[pid_arr >= 0]
                xyz_arr = np.zeros((pid_arr.shape[0], 3))
                for idx, pid in enumerate(pid_arr):
                    xyz_arr[idx] = self.recon_points[pid].xyz
                self.image_id2points[img_id] = xyz_arr
            self.xyz_arr = np.zeros((4, 3))
        elif "Cambridge" in str(self.options.scene):
            self.ds_type = "Cambridge"
            _logger.info(
                f"Reading sfm from {self.options.scene / 'reconstruction.nvm'}"
            )
            self.xyz_arr, self.image2points, self.image2name = read_nvm_file(
                self.options.scene / "reconstruction.nvm"
            )

            self.name2id = {v: k for k, v in self.image2name.items()}

        a_dir = self.options.scene / "train"
        self.dataset = CamLocDataset(
            root_dir=a_dir,
            sfm_model_dir=self.sfm_model_dir,
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,  # Optional clustering for Cambridge experiments.
        )

        _logger.info(
            f"Training with constraint masks radius={self.options.sampling_radius}"
        )
        _logger.info(f"Using {a_dir}")
        _logger.info(
            "Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
                self.options.scene,
                len(self.dataset),
                self.dataset.mean_cam_center[0],
                self.dataset.mean_cam_center[1],
                self.dataset.mean_cam_center[2],
            )
        )

        self.regressor = Head(
            self.dataset.mean_cam_center,
            self.options.num_head_blocks,
            self.options.use_homogeneous,
            in_channels=128,
        )
        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor_dummy = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor_dummy = self.regressor_dummy.to(self.device)
        self.regressor_dummy.eval()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(
            self.regressor.parameters(), lr=self.options.learning_rate_min
        )

        # Setup learning rate scheduler.
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = (
            self.options.epochs
            * self.options.training_buffer_size
            // self.options.batch_size
        )
        self.iterations_output = 100  # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == "circle"),
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = {}
        self.error_tracker = {}
        self.uv_arr = None

        self.name2uv = {}

        conf = {
            "r2d2": {
                "output": "feats-r2d2-n5000-r1024",
                "model": {
                    "name": "r2d2",
                    "max_keypoints": 5000,
                },
                "preprocessing": {
                    "grayscale": False,
                    "resize_max": 1024,
                },
            },
            "netvlad": {
                "output": "global-feats-netvlad",
                "model": {"name": "netvlad"},
                "preprocessing": {"resize_max": 1024},
            },
        }

        default_conf = {
            "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
            "grayscale": False,
            "resize_max": None,
            "resize_force": False,
            "interpolation": "cv2_area",  # pil_linear is more accurate but slower
        }
        model_dict = conf["r2d2"]["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        self.encoder = Model(model_dict).eval().to(device)

        conf_ns = SimpleNamespace(**{**default_conf, **conf})
        conf_ns.grayscale = conf["r2d2"]["preprocessing"]["grayscale"]
        conf_ns.resize_max = conf["r2d2"]["preprocessing"]["resize_max"]
        self.conf = conf_ns

        self.encoder_global = VPRModel(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 64,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 2,
            },
        ).cuda()

        state_dict = torch.load(
            "../MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt"
        )
        self.encoder_global.load_state_dict(state_dict)
        self.encoder_global.eval()
        self.image2desc = self.collect_image_descriptors()
        self.pid2mean_desc, self.all_pid_in_train_set = self.collect_descriptors(
            conf_ns
        )

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """
        # self.vis_kp()
        # self.collect_points_from_pretrained()
        # self.collect_descriptors()

        creating_buffer_time = 0.0
        training_time = 0.0

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(
            f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s."
        )

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time

        end_time = time.time()
        _logger.info(
            f"Done without errors. "
            f"Creating buffer time: {creating_buffer_time:.1f} seconds. "
            f"Training time: {training_time:.1f} seconds. "
            f"Total time: {end_time - self.training_start:.1f} seconds."
        )
        self.save_model()

    def collect_image_descriptors(self):
        testset = CamLocDataset(
            self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=self.options.image_resolution,
        )
        image2desc = {}
        with torch.no_grad():
            for example in tqdm(testset, desc="Collecting image descriptors"):
                image = load_image_mix_vpr(example[-4])
                image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                image_descriptor = image_descriptor.squeeze().cpu().numpy()
                image2desc[example[-4]] = image_descriptor
        return image2desc

    def collect_descriptors(self, conf):
        file_name1 = f"output/{self.ds_name}/codebook_r2d2.npy"
        file_name2 = f"output/{self.ds_name}/all_pids_r2d2.npy"
        if os.path.isfile(file_name1):
            pid2mean_desc = np.load(file_name1)
            all_pid = np.load(file_name2)
        else:
            testset = CamLocDataset(
                self.options.scene / "train",
                mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
                image_height=self.options.image_resolution,
            )
            pid2descriptors = {}
            with torch.no_grad():
                for example in tqdm(testset, desc="Collect point descriptors"):
                    image = read_and_preprocess(example[-4], conf)

                    pred = self.encoder(
                        {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                    )

                    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                    image_descriptor = self.image2desc[example[-4]]

                    keypoints = pred["keypoints"]
                    descriptors = pred["descriptors"].T

                    selected_pid, mask, ind = self.retrieve_pid(example, keypoints)
                    idx_arr, ind2 = np.unique(ind[mask], return_index=True)
                    selected_descriptors = descriptors[idx_arr]
                    selected_descriptors = 0.5 * (
                        selected_descriptors + image_descriptor
                    )

                    for idx, pid in enumerate(selected_pid[ind2]):
                        pid2descriptors.setdefault(pid, []).append(
                            selected_descriptors[idx]
                        )

            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            desc_dim = pid2descriptors[list(pid2descriptors.keys())[0]][0].shape[0]
            pid2mean_desc = np.zeros(
                (self.xyz_arr.shape[0], desc_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]][0].dtype,
            )

            for pid in pid2descriptors:
                pid2mean_desc[pid] = np.mean(pid2descriptors[pid], 0)

            np.save(file_name1, pid2mean_desc)
            np.save(file_name2, all_pid)

        return pid2mean_desc, all_pid

    def retrieve_pid(self, example, keypoints):
        pid_list = self.retrieve_pid_list_for_an_image(example[9])
        xyz = self.xyz_arr[pid_list]
        xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyzt = torch.from_numpy(xyzt).permute([1, 0]).float()
        gt_inv_pose_34 = example[4][:3]
        cam_coords = torch.mm(gt_inv_pose_34, xyzt)
        uv = torch.mm(example[6], cam_coords)
        uv[2].clamp_(min=0.1)  # avoid division by zero
        uv = uv[0:2] / uv[2]
        uv = uv.permute([1, 0]).cpu().numpy()

        tree = KDTree(keypoints)
        dis, ind = tree.query(uv)
        mask = dis < 5
        selected_pid = np.array(pid_list)[mask]
        return selected_pid, mask, ind

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(self.dataset, generator=self.batch_generator),
            batch_size=1,
            drop_last=False,
        )

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        training_dataloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            "features": torch.empty(
                (self.options.training_buffer_size, self.regressor.feature_dim),
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=self.device,
            ),
            "target_px": torch.empty(
                (self.options.training_buffer_size, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses_inv": torch.empty(
                (self.options.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses": torch.empty(
                (self.options.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics_inv": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        pbar = tqdm(total=self.options.training_buffer_size, desc="Filling buffer")
        count = 0
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                for (
                    image_B1HW,
                    image_ori,
                    image_mask_B1HW,
                    gt_pose_B44,
                    gt_pose_inv_B44,
                    gt_pose_sfm_inv_B44,
                    intrinsics_B33,
                    intrinsics_inv_B33,
                    _,
                    frame_path,
                    image_id_from_ds,
                    angle,
                    scale_factor,
                ) in training_dataloader:
                    count += 1

                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(
                        self.device, non_blocking=True
                    )

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor_dummy.get_features(image_B1HW)

                    B, C, H, W = features_BCHW.shape

                    mask_from_pretrained, pid_list = self.retrieve_gt_xyz(
                        image_ori,
                        frame_path,
                        gt_pose_inv_B44,
                        intrinsics_B33,
                        H,
                        W,
                    )

                    if torch.sum(mask_from_pretrained) < 10:
                        continue

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(
                        image_mask_B1HW,
                        [H, W],
                        interpolation=TF.InterpolationMode.NEAREST,
                    )

                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[
                        :, :H, :W
                    ].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(
                        B, 2, H, W
                    )  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = (
                        gt_pose_inv.unsqueeze(1)
                        .expand(B, H * W, 3, 4)
                        .reshape(-1, 3, 4)
                    )

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = (
                        intrinsics_B33.unsqueeze(1)
                        .expand(B, H * W, 3, 3)
                        .reshape(-1, 3, 3)
                    )
                    intrinsics_inv = (
                        intrinsics_inv_B33.unsqueeze(1)
                        .expand(B, H * W, 3, 3)
                        .reshape(-1, 3, 3)
                    )

                    batch_data = {
                        "features": normalize_shape(features_BCHW),
                        "target_px": normalize_shape(pixel_positions_B2HW),
                        "gt_poses_inv": gt_pose_inv,
                        "intrinsics": intrinsics,
                        "intrinsics_inv": intrinsics_inv,
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)
                    image_mask_N1 = image_mask_N1 * mask_from_pretrained.unsqueeze(1)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(
                        features_to_select,
                        self.options.training_buffer_size - buffer_idx,
                    )

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        features_to_select,
                        replacement=False
                        if features_to_select < torch.sum(image_mask_N1).item()
                        else True,
                        generator=self.sampling_generator,
                    )
                    oracular_desc = torch.from_numpy(
                        self.pid2mean_desc[pid_list]
                    ).cuda()
                    batch_data["features"] = oracular_desc

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[
                            k
                        ]

                    buffer_idx = buffer_offset
                    pbar.update(features_to_select)
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum(
            [v.element_size() * v.nelement() for k, v in self.training_buffer.items()]
        )
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(
            f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data."
        )
        self.regressor.train()

    def retrieve_gt_xyz(
        self,
        image_ori,
        frame_path,
        gt_pose_inv_B44,
        intrinsics_B33,
        H,
        W,
    ):
        batch_idx = 0
        xyz = None
        if self.ds_type == "Cambridge":
            pid_list = self.retrieve_pid_list_for_an_image(frame_path[batch_idx])
            xyz = self.xyz_arr[pid_list]
        elif (
            self.ds_type == "7scenes"
            or self.ds_type == "12scenes"
            or self.ds_type == "wayspots"
        ):
            base_key = frame_path[0].split("/")[-1]
            xyz = self.image_id2points[self.image_name2id[base_key]]

        xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyzt = torch.from_numpy(xyzt).permute([1, 0]).float().cuda()

        gt_inv_pose_34 = gt_pose_inv_B44[0, :3]
        cam_coords = torch.mm(gt_inv_pose_34, xyzt)
        uv = torch.mm(intrinsics_B33[0], cam_coords)
        uv[2].clamp_(min=0.1)  # avoid division by zero
        uv = uv[0:2] / uv[2]
        uv = uv.permute([1, 0]).cpu().numpy()

        uv_grid = self.pixel_grid_2HW[:, :H, :W].clone()
        uv_grid_arr = uv_grid.view(2, -1).permute([1, 0]).cpu().numpy()

        b1, b2 = np.max(uv_grid_arr, 0)
        oob_mask1 = np.bitwise_and(0 <= uv[:, 0], uv[:, 0] < b1)
        oob_mask2 = np.bitwise_and(0 <= uv[:, 1], uv[:, 1] < b2)
        oob_mask = np.bitwise_and(oob_mask1, oob_mask2)

        if np.sum(oob_mask) == 0:
            return None
        tree = KDTree(uv[oob_mask].astype(int))
        dis, ind = tree.query(uv_grid_arr)
        mask = dis < 5

        pid_list = np.array(pid_list)
        pid_list = pid_list[oob_mask][ind]

        mask = mask.astype(int)
        mask = torch.from_numpy(mask).cuda()
        return mask, pid_list

    def retrieve_pid_list_for_an_image(self, img_name):
        base_key = "/".join(img_name.split("/")[-1].split(".png")[0].split("_"))
        image_key1 = f"{base_key}.jpg"
        image_id_from_map = None
        if image_key1 in self.name2id:
            image_id_from_map = self.name2id[image_key1]
        else:
            image_key2 = f"{base_key}.png"
            if image_key2 in self.name2id:
                image_id_from_map = self.name2id[image_key2]

        pid_list = self.image2points[image_id_from_map]
        return pid_list

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(
            self.training_buffer["features"].shape[0], generator=self.training_generator
        )

        # Iterate with mini batches.
        buffer_size = self.training_buffer["features"].shape[0]
        for batch_start in range(0, buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer["features"][random_batch_indices].contiguous(),
                self.training_buffer["target_px"][random_batch_indices].contiguous(),
                self.training_buffer["gt_poses_inv"][random_batch_indices].contiguous(),
                self.training_buffer["intrinsics"][random_batch_indices].contiguous(),
                self.training_buffer["intrinsics_inv"][
                    random_batch_indices
                ].contiguous(),
            )

            self.iteration += 1

    def training_step(
        self,
        features_bC,
        target_px_b2,
        gt_inv_poses_b34,
        Ks_b33,
        invKs_b33,
    ):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = (
            features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        )

        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b3 = (
            pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).float()
        )

        pred_scene_coords_b31 = pred_scene_coords_b3.unsqueeze(-1)

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(
            reprojection_error_b2, dim=1, keepdim=True, p=1
        )

        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(
            valid_reprojection_error_b1, self.iteration
        )

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(
            invKs_b33, pixel_grid_crop_b31
        )

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = (
            torch.abs(target_camera_coords_b31 - pred_cam_coords_b31)
            .masked_select(invalid_mask_b11)
            .sum()
        )

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size
        # loss += mse_loss

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)

            _logger.info(
                f"Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, "
                f"Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s"
            )

        # Only step if the optimizer stepped and if we're not
        # over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        head_state_dict = self.regressor.state_dict()
        torch.save(head_state_dict, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")
        _logger.info(f"Finished training for {str(self.options.scene)}")
        self.test_model(self.regressor)

    def legal_predict(
        self,
        model,
        uv_arr,
        features_ori,
        gpu_index_flat,
        structured=True,
        remove_duplicate=False,
    ):
        distances, feature_indices = gpu_index_flat.search(features_ori, 1)

        feature_indices = feature_indices.ravel()
        features_ora = (
            torch.from_numpy(
                self.pid2mean_desc[self.all_pid_in_train_set][feature_indices]
            )
            .float()
            .cuda()
        )

        # mask = np.isin(self.all_pid_in_train_set[feature_indices], self.good_pids)

        if not structured:
            pred_scene_coords_b3 = predict_xyz(model, features_ora)
        else:
            if remove_duplicate:
                pid2uv = {}
                for idx in range(feature_indices.shape[0]):
                    pid = feature_indices[idx]
                    dis = distances[idx][0]
                    uv = uv_arr[idx]
                    if pid not in pid2uv:
                        pid2uv[pid] = [dis, uv]
                    else:
                        if dis < pid2uv[pid][0]:
                            pid2uv[pid] = [dis, uv]
                uv_arr = np.array([pid2uv[pid][1] for pid in pid2uv])
                feature_indices = [pid for pid in pid2uv]

            pred_scene_coords_b3 = self.xyz_arr[
                self.all_pid_in_train_set[feature_indices]
            ]

        return uv_arr, pred_scene_coords_b3

    def test_model(self, model=None):
        if model is None:
            print(f"Loaded weights from {self.options.output_map_file}")
            # Create regressor.
            model = Head(
                self.dataset.mean_cam_center,
                self.options.num_head_blocks,
                self.options.use_homogeneous,
                in_channels=128,
            )
            if os.path.exists(self.options.output_map_file):
                head_state_dict = torch.load(
                    self.options.output_map_file, map_location="cpu"
                )
                model.load_state_dict(head_state_dict)
            model = model.cuda()

        model.eval()
        testset = CamLocDataset(
            self.options.scene / "test",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=self.options.image_resolution,
            use_half=False,
        )
        testset_loader = DataLoader(testset, shuffle=False, num_workers=6)
        device = "cuda"

        rErrs = []
        tErrs = []

        pct10_5 = 0
        pct5 = 0
        pct2 = 0
        pct1 = 0
        mse_error = []
        index = faiss.IndexFlatL2(128)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_pid_in_train_set])

        with torch.no_grad():
            for (
                image_B1HW,
                image_ori,
                _,
                gt_pose_B44,
                gt_pose_inv_B44,
                _,
                intrinsics_B33,
                _,
                _,
                filenames,
                _,
                _,
                _,
            ) in tqdm(testset_loader):
                intrinsics_B33 = intrinsics_B33.to(device, non_blocking=True)

                image = read_and_preprocess(filenames[0], self.conf)
                pred = self.encoder(
                    {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                )
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                keypoints = pred["keypoints"]
                descriptors = pred["descriptors"].T

                intrinsics_33 = intrinsics_B33[0].cpu()
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                image = load_image_mix_vpr(filenames[0])
                image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                image_descriptor = image_descriptor.squeeze().cpu().numpy()

                descriptors = 0.5 * (descriptors + image_descriptor)

                uv_arr2, xyz_pred2 = self.legal_predict(
                    model,
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                    structured=True,
                )

                # uv_arr2, xyz_pred2 = self.predict_abs(model, keypoints, descriptors)
                t_err, r_err = compute_pose(
                    uv_arr2, xyz_pred2, focal_length, ppX, ppY, gt_pose_B44
                )

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Check various thresholds.
                if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                    pct1 += 1

        total_frames = len(rErrs)
        assert total_frames == len(testset)

        # Compute median errors.
        tErrs.sort()
        rErrs.sort()
        median_idx = total_frames // 2
        median_rErr = rErrs[median_idx]
        median_tErr = tErrs[median_idx]

        # Compute final metrics.
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100

        _logger.info("===================================================")
        _logger.info("Test complete.")

        _logger.info("Accuracy:")
        _logger.info(f"\t10cm/5deg: {pct10_5:.1f}%")
        _logger.info(f"\t5cm/5deg: {pct5:.1f}%")
        _logger.info(f"\t2cm/2deg: {pct2:.1f}%")
        _logger.info(f"\t1cm/1deg: {pct1:.1f}%")

        _logger.info(f"Median Error: {median_rErr}deg, {median_tErr}cm")
        _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
        if len(mse_error) > 0:
            _logger.info(f"MAE Error: {np.mean(mse_error)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = read_args()
    options = parser.parse_args()
    trainer = TrainerACE(options)
    trainer.test_model()
    # trainer.train()
