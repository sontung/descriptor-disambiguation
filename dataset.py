import logging
import math
import os
import pickle
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pycolmap
from PIL import Image
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from pykdtree.kdtree import KDTree
from skimage import color
from skimage import io
from skimage.transform import rotate, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm

from hloc.pipelines.RobotCar.pipeline import CONDITIONS, generate_query_list
import ace_util
import colmap_read
import dd_utils
from ace_network import Regressor
from skimage.transform import rotate as ski_rotate
from skimage.transform import resize as ski_resize
from os import listdir
from os.path import isfile, join
from ace_util import project_using_pose
import faiss

_logger = logging.getLogger(__name__)


class CamLocDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(
        self,
        root_dir,
        sfm_model_dir=None,
        mode=0,
        using_sfm_poses=True,
        sparse=False,
        augment=False,
        aug_rotation=15,
        aug_scale_min=2 / 3,
        aug_scale_max=3 / 2,
        aug_black_white=0.1,
        aug_color=0.3,
        image_height=480,
        use_half=True,
        num_clusters=None,
        cluster_idx=None,
    ):
        self.using_sfm_poses = True
        self.sfm_model_dir = sfm_model_dir

        self.use_half = use_half

        self.init = mode == 1
        self.sparse = sparse
        self.eye = mode == 2

        self.image_height = image_height

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        self.num_clusters = num_clusters
        self.cluster_idx = cluster_idx

        self.using_sfm_poses = False
        self.image_name2id = None
        if self.sfm_model_dir is not None:
            self.using_sfm_poses = True
            _logger.info(f"Reading SFM poses from {self.sfm_model_dir}")
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
            root_dir_str = str(root_dir)
            for image_id, image in self.recon_images.items():
                if "test" in image.name and "test" not in root_dir_str:
                    continue
                elif "train" in image.name and "train" not in root_dir_str:
                    continue
                if "wayspots" in str(root_dir):
                    self.image_name2id[image.name.split("/")[-1]] = image_id
                if "7scenes" in str(root_dir):
                    self.image_name2id[image.name.replace("/", "-")] = image_id

        if self.num_clusters is not None:
            if self.num_clusters < 1:
                raise ValueError("num_clusters must be at least 1")

            if self.cluster_idx is None:
                raise ValueError(
                    "cluster_idx needs to be specified when num_clusters is set"
                )

            if self.cluster_idx < 0 or self.cluster_idx >= self.num_clusters:
                raise ValueError(
                    f"cluster_idx needs to be between 0 and {self.num_clusters - 1}"
                )

        if (
            self.eye
            and self.augment
            and (
                self.aug_rotation > 0
                or self.aug_scale_min != 1
                or self.aug_scale_max != 1
            )
        ):
            # pre-generated eye coordinates cannot be augmented
            _logger.warning(
                "WARNING: Check your augmentation settings. Camera coordinates will not be augmented."
            )

        # Setup data paths.
        root_dir = Path(root_dir)

        # Main folders.
        rgb_dir = root_dir / "rgb"
        pose_dir = root_dir / "poses"
        calibration_dir = root_dir / "calibration"

        # Optional folders. Unused in ACE.
        if self.eye:
            coord_dir = root_dir / "eye"
        elif self.sparse:
            coord_dir = root_dir / "init"
        else:
            coord_dir = root_dir / "depth"

        # Find all images. The assumption is that it only contains image files.
        self.rgb_files = sorted(rgb_dir.iterdir())

        # Find all ground truth pose files. One per image.
        self.pose_files = sorted(pose_dir.iterdir())

        # Load camera calibrations. One focal length per image.
        self.calibration_files = sorted(calibration_dir.iterdir())

        if self.init or self.eye:
            # Load GT scene coordinates.
            self.coord_files = sorted(coord_dir.iterdir())
        else:
            self.coord_files = None

        if len(self.rgb_files) != len(self.pose_files):
            raise RuntimeError("RGB file count does not match pose file count!")

        if len(self.rgb_files) != len(self.calibration_files):
            raise RuntimeError("RGB file count does not match calibration file count!")

        if self.coord_files and len(self.rgb_files) != len(self.coord_files):
            raise RuntimeError("RGB file count does not match coordinate file count!")

        # Create grid of 2D pixel positions used when generating scene coordinates from depth.
        if self.init and not self.sparse:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.augment:
            self.image_transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ColorJitter(
                        brightness=self.aug_black_white, contrast=self.aug_black_white
                    ),
                    # saturation=self.aug_color, hue=self.aug_color),  # Disable colour augmentation.
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.4
                        ],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25],
                    ),
                ]
            )
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.4
                        ],  # statistics calculated over 7scenes training set, should generalize fairly well
                        std=[0.25],
                    ),
                ]
            )

        # We use this to iterate over all frames. If clustering is enabled this is used to filter them.
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # If clustering is enabled.
        if self.num_clusters is not None:
            _logger.info(
                f"Clustering the {len(self.rgb_files)} into {num_clusters} clusters."
            )
            _, _, cluster_labels = self._cluster(num_clusters)

            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            _logger.info(
                f"After clustering, chosen cluster: {cluster_idx}, Using {len(self.valid_file_indices)} images."
            )

        # Calculate mean camera center (using the valid frames only).
        self.mean_cam_center = self._compute_mean_camera_center()
        self.root_dir = str(root_dir)
        self.image_name2uv = None

    @staticmethod
    def _create_prediction_grid():
        # Assumes all input images have a resolution smaller than 5000x5000.
        prediction_grid = np.zeros(
            (
                2,
                math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
                math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
            )
        )

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * Regressor.OUTPUT_SUBSAMPLE
                prediction_grid[1, y, x] = y * Regressor.OUTPUT_SUBSAMPLE

        return prediction_grid

    @staticmethod
    def _resize_image(image, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode="constant", cval=0):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode, cval=cval)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _cluster(self, num_clusters):
        """
        Clusters the dataset using hierarchical kMeans.
        Initialization:
            Put all images in one cluster.
        Interate:
            Pick largest cluster.
            Split with kMeans and k=2.
            Input for kMeans is the 3D median scene coordiante per image.
        Terminate:
            When number of target clusters has been reached.
        Returns:
            cam_centers: For each cluster the mean (not median) scene coordinate
            labels: For each image the cluster ID
        """
        num_images = len(self.pose_files)
        _logger.info(
            f"Clustering a dataset with {num_images} frames into {num_clusters} clusters."
        )

        # A tensor holding all camera centers used for clustering.
        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i in range(num_images):
            pose = self._load_pose(i)
            cam_centers[i] = pose[:3, 3]

        # Setup kMEans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS

        # Label of next cluster.
        label_counter = 0

        # Initialise list of clusters with all images.
        clusters = []
        clusters.append((cam_centers, label_counter, np.zeros(3)))

        # All images belong to cluster 0.
        labels = np.zeros(num_images)

        # iterate kMeans with k=2
        while len(clusters) < num_clusters:
            # Select largest cluster (list is sorted).
            cur_cluster = clusters.pop(0)
            label_counter += 1

            # Split cluster.
            cur_error, cur_labels, cur_centroids = cv2.kmeans(
                cur_cluster[0], 2, None, criteria, 10, flags
            )

            # Update cluster list.
            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels

            # Sort updated list.
            clusters = sorted(
                clusters, key=lambda cluster: cluster[0].shape[0], reverse=True
            )

        # clusters are sorted but cluster indices are random, remap cluster indices to sorted indices
        remapped_labels = np.zeros(num_images)
        remapped_clusters = []

        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters

        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            # Compute distance of each cam to the center of the cluster.
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0

            # First compute the center of the cluster (mean).
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1

            cluster_centers[cluster[1]] = cam_data.mean(0)

            # Compute the distance of each cam from the cluster center. Then average and square.
            cam_dists = np.broadcast_to(
                cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3)
            )
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists ** 2

            cluster_sizes[cluster[1]] = cam_dists.mean()

            _logger.info(
                "Cluster %i: %.1fm, %.1fm, %.1fm, images: %i, mean squared dist: %f"
                % (
                    cluster[1],
                    cluster_centers[cluster[1]][0],
                    cluster_centers[cluster[1]][1],
                    cluster_centers[cluster[1]][2],
                    cluster[0].shape[0],
                    cluster_sizes[cluster[1]],
                )
            )

        _logger.info("Clustering done.")

        return cluster_centers, cluster_sizes, labels

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3,))

        for idx in self.valid_file_indices:
            pose = self._load_pose(idx)

            # Get the translation component.
            mean_cam_center += pose[0:3, 3]

        # Avg.
        mean_cam_center /= len(self)
        return mean_cam_center

    def _load_image(self, idx):
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def _load_pose(self, idx):
        # Stored as a 4x4 matrix.
        if self.using_sfm_poses:
            pose = self._load_pose_from_sfm(idx)
        else:
            pose = np.loadtxt(self.pose_files[idx])
            pose = torch.from_numpy(pose).float()
        return pose

    def _load_pose_from_sfm(self, idx):
        if self.image_name2id is None:
            return torch.ones((4, 4)).float()
        img_id = self.image_name2id[str(self.rgb_files[idx]).split("/")[-1]]
        qvec = self.recon_images[img_id].qvec
        tvec = self.recon_images[img_id].tvec
        pose = ace_util.return_pose_mat(qvec, tvec)
        pose = torch.from_numpy(pose).float()
        return pose

    def _get_single_item(self, idx, image_height):
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)

        # Load intrinsics.
        k = np.loadtxt(self.calibration_files[idx])
        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception(
                "Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera."
            )

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_height / image.shape[0]
        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        # Rescale image.
        image = self._resize_image(image, image_height)
        image_ori = np.copy(np.array(image))

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Apply remaining transforms.
        image = self.image_transform(image)

        # Load pose.
        pose = self._load_pose(idx)

        # Load ground truth scene coordinates, if needed.
        if self.init:
            if self.sparse:
                coords = torch.load(self.coord_files[idx])
            else:
                depth = io.imread(self.coord_files[idx])
                depth = depth.astype(np.float64)
                depth /= 1000  # from millimeters to meters
        elif self.eye:
            coords = torch.load(self.coord_files[idx])
        else:
            coords = 0  # Default for ACE, we don't need them.

        # Apply data augmentation if necessary.
        angle_deg = 0
        if self.augment:
            # Generate a random rotation angle.
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # Rotate input image and mask.
            image = self._rotate_image(image, angle, 1, "reflect")
            image_ori = rotate(
                image_ori, angle, order=1, mode="reflect", preserve_range=True
            )
            image_mask = self._rotate_image(image_mask, angle, order=1, mode="constant")
            # semantic_mask = torch.from_numpy(semantic_mask).unsqueeze(0)
            # semantic_mask = self._rotate_image(semantic_mask, angle, order=1, mode="constant", cval=2)

            # If we loaded the GT scene coordinates.
            if self.init:
                if self.sparse:
                    # rotate and scale initalization targets
                    coords_w = math.ceil(image.size(2) / Regressor.OUTPUT_SUBSAMPLE)
                    coords_h = math.ceil(image.size(1) / Regressor.OUTPUT_SUBSAMPLE)
                    coords = F.interpolate(
                        coords.unsqueeze(0), size=(coords_h, coords_w)
                    )[0]

                    coords = self._rotate_image(coords, angle, 0)
                else:
                    # rotate and scale depth maps
                    depth = resize(depth, image.shape[1:], order=0)
                    depth = rotate(depth, angle, order=0, mode="constant")

            # Rotate ground truth camera pose as well.
            angle_deg = angle
            angle = angle * math.pi / 180.0
            # Create a rotation matrix.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)

            # Apply rotation matrix to the ground truth camera pose.
            pose = torch.matmul(pose, pose_rot)

        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose.
        pose_inv = pose.inverse()
        # if self.image_name2id is not None:
        #     pose_sfm_inv = pose_sfm.inverse()
        # else:
        #     pose_sfm_inv = pose_sfm
        pose_sfm_inv = pose_inv

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)

        # Hardcode the principal point to the centre of the image unless otherwise specified.
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image.shape[2] / 2  # 427
            intrinsics[1, 2] = image.shape[1] / 2  # 240

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()

        return (
            image,
            image_ori,
            image_mask,
            # new_mask,
            pose,
            pose_inv,
            pose_sfm_inv,
            intrinsics,
            intrinsics_inv,
            coords,
            str(self.rgb_files[idx]),
            idx,
            angle_deg,
            f_scale_factor,
        )

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_height = int(self.image_height * scale_factor)

        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i, image_height) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_height)


def read_intrinsic(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    name2params = {}
    for line in lines:
        img_name, cam_type, w, h, f, cx, cy, k = line.split(" ")
        f, cx, cy, k = map(float, [f, cx, cy, k])
        w, h = map(int, [w, h])
        name2params[img_name] = [cam_type, w, h, f, cx, cy, k]
    return name2params


class AachenDataset(Dataset):
    def __init__(self, ds_dir="datasets/aachen_v1.1", train=True):
        self.ds_type = "aachen"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/3D-models/aachen_v_1_1"
        self.images_dir_str = f"{self.ds_dir}/images_upright"
        self.images_dir = Path(self.images_dir_str)

        self.train = train
        self.day_intrinsic_file = (
            f"{self.ds_dir}/queries/day_time_queries_with_intrinsics.txt"
        )
        self.night_intrinsic_file = (
            f"{self.ds_dir}/queries/night_time_queries_with_intrinsics.txt"
        )

        if self.train:
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
                self.image_name2id[image.name] = image_id
            self.image_id2points = {}
            self.pid2images = {}

            for img_id in tqdm(self.recon_images):
                pid_arr = self.recon_images[img_id].point3D_ids
                pid_arr = pid_arr[pid_arr >= 0]
                xyz_arr = np.zeros((pid_arr.shape[0], 3))
                for idx, pid in enumerate(pid_arr):
                    xyz_arr[idx] = self.recon_points[pid].xyz
                    # self.pid2images.setdefault(pid, []).append(img_id)
                self.image_id2points[img_id] = xyz_arr
            # self.im_names = list(self.image_name2id.keys())
            self.img_ids = list(self.image_name2id.values())
        else:
            name2params1 = read_intrinsic(self.day_intrinsic_file)
            name2params2 = read_intrinsic(self.night_intrinsic_file)
            self.name2params = {**name2params1, **name2params2}
            self.img_ids = list(self.name2params.keys())
        return

    def _load_image(self, img_id):
        name = self.recon_images[img_id].name
        name2 = str(self.images_dir / name)
        image = io.imread(name2)

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image, name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]

            image, image_name = self._load_image(img_id)
            camera_id = self.recon_images[img_id].camera_id
            camera = self.recon_cameras[camera_id]
            focal, cx, cy, k = camera.params
            intrinsics = torch.eye(3)

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            qvec = self.recon_images[img_id].qvec
            tvec = self.recon_images[img_id].tvec
            # pose = utils.return_pose_mat(qvec, tvec)
            pose_inv = dd_utils.return_pose_mat_no_inv(qvec, tvec)

            xyz_gt = self.image_id2points[img_id]
            pid_list = self.recon_images[img_id].point3D_ids
            mask = pid_list >= 0
            pid_list = pid_list[mask]
            uv_gt = self.recon_images[img_id].xys[mask]

            pose_inv = torch.from_numpy(pose_inv)

        else:
            name1 = self.img_ids[idx]
            image_name = str(self.images_dir / name1)

            cam_type, width, height, focal, cx, cy, k = self.name2params[name1]
            camera = pycolmap.Camera(
                model=cam_type,
                width=int(width),
                height=int(height),
                params=[focal, cx, cy, k],
            )

            intrinsics = torch.eye(3)

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            image = None
            img_id = name1
            pid_list = []
            pose_inv = None
            xyz_gt = None
            uv_gt = None

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            intrinsics,
            camera,
            xyz_gt,
            uv_gt,
        )

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


def _read_train_poses(a_file):
    with open(a_file) as file:
        lines = [line.rstrip() for line in file]
    name2mat = {}
    for line in lines:
        img_name, *matrix = line.split(" ")
        if matrix:
            matrix = np.array(matrix, float).reshape(4, 4)
        name2mat[img_name] = matrix
    return name2mat


def _produce_image_descriptor(name2, conf_ns_retrieval, encoder_global):
    image, _ = ace_util.read_and_preprocess(name2, conf_ns_retrieval)
    image_descriptor = (
        encoder_global({"image": torch.from_numpy(image).unsqueeze(0).cuda()})[
            "global_descriptor"
        ]
        .squeeze()
        .cpu()
        .numpy()
    )
    return image_descriptor


class RobotCarDataset(Dataset):
    def __init__(self, ds_dir="datasets/robotcar", train=True, evaluate=False):
        self.ds_type = "robotcar"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/3D-models/all-merged/all.nvm"
        self.images_dir = Path(f"{self.ds_dir}/images")
        self.test_file1 = f"{ds_dir}/robotcar_v2_train.txt"
        self.test_file2 = f"{ds_dir}/robotcar_v2_test.txt"
        self.ds_dir_path = Path(self.ds_dir)
        self.images_dir_str = str(self.images_dir)
        self.train = train
        self.evaluate = evaluate
        if evaluate:
            assert not self.train

        if self.train:
            (
                self.xyz_arr,
                self.image2points,
                self.image2name,
                self.image2pose,
                self.image2info,
                self.image2uvs,
            ) = ace_util.read_nvm_file(self.sfm_model_dir)
            self.name2image = {v: k for k, v in self.image2name.items()}
            self.img_ids = list(self.image2name.keys())

            self.name2mat = _read_train_poses(self.test_file1)

            # start_id = max(self.img_ids)+1
            # self.name2id = {}
            # for name in self.name2mat:
            #     pose = self.name2mat[name]
            #     self.name2id[name] = start_id
            #     self.image2name[start_id] = f"./{name}"
            #     self.image2pose[start_id] = pose
            #     self.img_ids.append(start_id)
            #     start_id += 1
            # self.complete_image2points()

            # self.img_ids = self.img_ids[-len(self.name2mat):]
            # import open3d as o3d
            # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr))
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1025)
            # vis.add_geometry(point_cloud)
            # vis.run()
            # vis.destroy_window()
        else:
            self.ts2cond = {}
            for condition in CONDITIONS:
                all_image_names = list(Path.glob(self.images_dir, f"{condition}/*/*"))

                for name in all_image_names:
                    time_stamp = str(name).split("/")[-1].split(".")[0]
                    self.ts2cond.setdefault(time_stamp, []).append(condition)
            for ts in self.ts2cond:
                assert len(self.ts2cond[ts]) == 3

            if not self.evaluate:
                self.name2mat = _read_train_poses(self.test_file1)
            else:
                self.name2mat = _read_train_poses(self.test_file2)
            self.img_ids = list(self.name2mat.keys())

        return

    def process_pid_list(self, pose_mat, intrinsics, tree, inverse=True):
        if not inverse:
            uv_gt = project_using_pose(
                torch.from_numpy(pose_mat).unsqueeze(0).cuda().float(),
                intrinsics.unsqueeze(0).cuda().float(),
                self.xyz_arr,
            )
        else:
            uv_gt = project_using_pose(
                torch.from_numpy(pose_mat).inverse().unsqueeze(0).cuda().float(),
                intrinsics.unsqueeze(0).cuda().float(),
                self.xyz_arr,
            )
        oob_mask1 = np.bitwise_and(0 <= uv_gt[:, 0], uv_gt[:, 0] < 1024)
        oob_mask2 = np.bitwise_and(0 <= uv_gt[:, 1], uv_gt[:, 1] < 1024)
        oob_mask = np.bitwise_and(oob_mask1, oob_mask2)
        dis, ind = tree.query(uv_gt[oob_mask])
        mask = dis < 5
        pid_list = np.arange(self.xyz_arr.shape[0])[oob_mask][mask]
        return pid_list

    def complete_image2points(self):
        # retrieval_model = "netvlad"
        # global_feature_dim = 2048
        # conf, default_conf = dd_utils.hloc_conf_for_all_models()
        #
        # model_dict = conf[retrieval_model]["model"]
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # Model = dynamic_load(extractors, model_dict["name"])
        # encoder_global = Model(model_dict).eval().to(device)
        # conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        # conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]
        #
        # file_name1 = (
        #     f"output/{self.ds_type}/small_image_desc_{retrieval_model}.npy"
        # )
        # file_name2 = (
        #     f"output/{self.ds_type}/small_image_desc_ids_{retrieval_model}.npy"
        # )
        # if os.path.isfile(file_name1):
        #     all_desc = np.load(file_name1)
        #     afile = open(file_name2, "rb")
        #     all_ids = pickle.load(afile)
        #     afile.close()
        # else:
        #     all_desc = np.zeros((len(self.image2points), global_feature_dim))
        #     all_ids = []
        #     idx = 0
        #     with torch.no_grad():
        #         for img_id in tqdm(self.image2points, desc="Collecting image descriptors for reference images"):
        #             name2 = self._process_id_to_name(img_id)
        #             image_descriptor = _produce_image_descriptor(name2, conf_ns_retrieval, encoder_global)
        #             all_desc[idx] = image_descriptor
        #             all_ids.append(img_id)
        #             idx += 1
        #     np.save(file_name1, all_desc)
        #     with open(file_name2, "wb") as handle:
        #         pickle.dump(all_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # all_desc_train = np.zeros((len(self.name2mat), global_feature_dim))
        # all_ids_train = []
        # idx = 0
        # with torch.no_grad():
        #     for img_id in tqdm(self.img_ids, desc="Collecting image descriptors for training images"):
        #         if img_id not in self.image2points:
        #             name2 = self._process_id_to_name(img_id)
        #             image_descriptor = _produce_image_descriptor(name2, conf_ns_retrieval, encoder_global)
        #             all_desc_train[idx] = image_descriptor
        #             all_ids_train.append(img_id)
        #             idx += 1
        # index = faiss.IndexFlatL2(global_feature_dim)  # build the index
        # res = faiss.StandardGpuResources()
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        # gpu_index_flat.add(all_desc)
        # distances, image_indices = gpu_index_flat.search(all_desc_train, 10)
        # for idx, img_id in enumerate(all_ids_train):
        #     images_retrieved = image_indices[idx]
        #     id_retrieved = [all_ids[id_] for id_ in images_retrieved]
        #     points = []
        #     for id_ in id_retrieved:
        #         points_ref = self.image2points[id_]
        #         points.extend(points_ref)
        #     self.image2points[img_id] = list(set(points))

        features_path = f"output/robotcar/r2d2_features_train.h5"
        features_h5 = h5py.File(features_path, "r")
        du0 = 0
        pkl_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/RobotCar_hloc_superpoint+superglue_netvlad20.txt_logs.pkl"
        afile = open(pkl_file, "rb")
        data_hloc = pickle.load(afile)
        afile.close()

        hloc_sfm_model = colmap_read.read_points3D_binary(
            "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/sfm_superpoint+superglue/points3D.bin"
        )
        tree = KDTree(self.xyz_arr)

        for name in tqdm(self.name2mat, desc="Completing training images"):
            keypoints, descriptors = dd_utils.read_kp_and_desc(
                f"{self.images_dir_str}/{name}", features_h5
            )
            tree = KDTree(keypoints)
            focal = 400
            if "rear" in name:
                cx = 508.222931
                cy = 498.187378
            elif "right" in name:
                cx = 502.503754
                cy = 490.259033
            elif "left" in name:
                cx = 500.107605
                cy = 511.461426
            img_id = self.name2id[name]

            pose_mat = self.image2pose[img_id]
            intrinsics = torch.eye(3)
            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            pid_list = self.process_pid_list(pose_mat, intrinsics, tree, inverse=False)

            self.image2points[img_id] = pid_list

            loc_res = data_hloc["loc"][name]
            img_id = self.name2id[name]

            intrinsics = torch.eye(3)
            focal, cx, cy, _ = loc_res["PnP_ret"]["camera"].params
            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            xyz_gt = np.array(
                [hloc_sfm_model[pid].xyz for pid in loc_res["points3D_ids"]]
            )

            tree = KDTree(self.xyz_arr)
            uv_gt = project_using_pose(
                torch.from_numpy(pose_mat).inverse().unsqueeze(0).cuda().float(),
                intrinsics.unsqueeze(0).cuda().float(),
                xyz_gt,
            )
            diff = np.mean(np.abs(loc_res["keypoints_query"] - uv_gt), 1)
            mask = diff < 5
            xyz_gt_ref = xyz_gt[mask]
            _, pid_list = tree.query(xyz_gt_ref)
            self.image2points[img_id] = pid_list

            import open3d as o3d

            point_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.xyz_arr[pid_list])
            )
            point_cloud2 = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(xyz_gt[mask])
            )
            point_cloud.paint_uniform_color((1, 0, 0))
            point_cloud2.paint_uniform_color((0, 1, 0))
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1920, height=1025)
            vis.add_geometry(point_cloud)
            vis.add_geometry(point_cloud2)
            vis.run()
            vis.destroy_window()

            # image = cv2.imread(str(self.images_dir/name))
            # for u, v in uv_gt[oob_mask][mask].astype(int):
            #     cv2.circle(image, (u, v), 5, (255, 0, 0))
            #
            # uv_gt = project_using_pose(
            #     torch.from_numpy(pose_mat).inverse().unsqueeze(0).cuda().float(),
            #     intrinsics.unsqueeze(0).cuda().float(),
            #     self.xyz_arr,
            # )
            # oob_mask1 = np.bitwise_and(0 <= uv_gt[:, 0], uv_gt[:, 0] < 1024)
            # oob_mask2 = np.bitwise_and(0 <= uv_gt[:, 1], uv_gt[:, 1] < 1024)
            # oob_mask = np.bitwise_and(oob_mask1, oob_mask2)
            # dis, ind = tree.query(uv_gt[oob_mask])
            # mask = dis < 1
            # for u, v in uv_gt[oob_mask][mask].astype(int):
            #     cv2.circle(image, (u, v), 5, (0, 255, 0))
            # cv2.imwrite(f"debug/test{du0}.png", image)
            # du0 += 1
            #
            # pid_list1 = self.process_pid_list(pose_mat, intrinsics, tree)
            # pid_list2 = self.process_pid_list(pose_mat, intrinsics, tree, inverse=False)
            # import open3d as o3d
            # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr[pid_list1]))
            # point_cloud2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr[pid_list2]))
            # point_cloud.paint_uniform_color((1, 0, 0))
            # point_cloud2.paint_uniform_color((0, 1, 0))
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1025)
            # vis.add_geometry(point_cloud)
            # vis.add_geometry(point_cloud2)
            # vis.run()
            # vis.destroy_window()

            # camera = pycolmap.Camera(
            #     model="SIMPLE_RADIAL",
            #     width=1024,
            #     height=1024,
            #     params=[focal, cx, cy, 0],
            # )
            # res = pycolmap.absolute_pose_estimation(
            #     uv_gt[oob_mask][mask],
            #     self.xyz_arr[pid_list],
            #     camera,
            #     # refinement_options={"max_num_iterations": 100},
            # )
            # pose_mat = torch.from_numpy(self.image2pose[img_id])
            # t_err = float(torch.norm(pose_mat[0:3, 3] - res["cam_from_world"].translation))
            # print(t_err)

        # pkl_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/RobotCar_hloc_superpoint+superglue_netvlad20.txt_logs.pkl"
        # afile = open(pkl_file, "rb")
        # data_hloc = pickle.load(afile)
        # afile.close()
        #
        # hloc_sfm_model = colmap_read.read_points3D_binary("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/sfm_superpoint+superglue/points3D.bin")
        # tree = KDTree(self.xyz_arr)
        # for name in self.name2mat:
        #     loc_res = data_hloc["loc"][name]
        #     img_id = self.name2id[name]
        #
        #     pose_mat = self.image2pose[img_id]
        #     intrinsics = torch.eye(3)
        #     focal, cx, cy, _ = loc_res["PnP_ret"]["camera"].params
        #     intrinsics[0, 0] = focal
        #     intrinsics[1, 1] = focal
        #     intrinsics[0, 2] = cx
        #     intrinsics[1, 2] = cy
        #     xyz_gt = np.array([hloc_sfm_model[pid].xyz for pid in loc_res["points3D_ids"]])
        #     uv_gt = project_using_pose(
        #         torch.from_numpy(pose_mat).unsqueeze(0).cuda().float(),
        #         intrinsics.unsqueeze(0).cuda().float(),
        #         xyz_gt,
        #     )
        #     diff = np.mean(np.abs(loc_res["keypoints_query"]-uv_gt), 1)
        #     mask = diff < 5
        #     xyz_gt_ref = xyz_gt[mask]
        #     _, pid_list = tree.query(xyz_gt_ref)
        #     self.image2points[img_id] = pid_list

    def _process_id_to_name(self, img_id):
        name = self.image2name[img_id].split("./")[-1]
        name2 = str(self.images_dir / name).replace(".png", ".jpg")
        return name2

    def _load_image(self, img_id):
        name2 = self._process_id_to_name(img_id)
        image = io.imread(name2)

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image, name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]
            image, image_name = self._load_image(img_id)
            if type(self.image2pose[img_id]) == list:
                qw, qx, qy, qz, tx, ty, tz = self.image2pose[img_id]
                pose_mat = dd_utils.return_pose_mat_no_inv(
                    [qw, qx, qy, qz], [tx, ty, tz]
                )
            else:
                pose_mat = self.image2pose[img_id]
                # pose_mat = np.linalg.inv(pose_mat)

            intrinsics = torch.eye(3)
            if img_id in self.image2info:
                focal, radial = self.image2info[img_id]
                cx, cy = 512, 512
            else:
                focal = 400
                if "rear" in image_name:
                    cx = 508.222931
                    cy = 498.187378
                elif "right" in image_name:
                    cx = 502.503754
                    cy = 490.259033
                elif "left" in image_name:
                    cx = 500.107605
                    cy = 511.461426

            assert image.shape == (1024, 1024, 3)
            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy

            pid_list = self.image2points[img_id]
            xyz_gt = self.xyz_arr[pid_list]

            # uv_gt = project_using_pose(
            #     torch.from_numpy(pose_mat).unsqueeze(0).cuda().float(),
            #     intrinsics.unsqueeze(0).cuda().float(),
            #     xyz_gt,
            # )

            uv_gt = np.array(self.image2uvs[img_id])
            camera = pycolmap.Camera(
                model="SIMPLE_RADIAL",
                width=1024,
                height=1024,
                params=[focal, cx, cy, 0],
            )

            pose_inv = torch.from_numpy(pose_mat)

        else:
            name0 = self.img_ids[idx]

            if self.evaluate:
                time_stamp = str(name0).split("/")[-1].split(".")[0]
                cond = self.ts2cond[time_stamp][0]
                name1 = f"{cond}/{name0}"
                if ".png" in name1:
                    name1 = name1.replace(".png", ".jpg")
            else:
                name1 = name0

            image_name = str(self.images_dir / name1)

            focal = 400
            if "rear" in name1:
                cx = 508.222931
                cy = 498.187378
            elif "right" in name1:
                cx = 502.503754
                cy = 490.259033
            elif "left" in name1:
                cx = 500.107605
                cy = 511.461426

            camera = pycolmap.Camera(
                model="SIMPLE_RADIAL",
                width=1024,
                height=1024,
                params=[focal, cx, cy, 0],
            )

            intrinsics = torch.eye(3)

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            image = None
            img_id = name1
            pid_list = []
            if type(self.name2mat[name0]) == np.ndarray:
                pose_inv = torch.from_numpy(self.name2mat[name0]).inverse()
            else:
                pose_inv = None
            xyz_gt = None
            uv_gt = None

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            intrinsics,
            camera,
            xyz_gt,
            uv_gt,
        )

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


class CMUDataset(Dataset):
    def __init__(self, ds_dir="datasets/datasets/cmu_extended/slice2", train=True):
        self.ds_type = f"cmu/{ds_dir.split('/')[-1]}"
        self.ds_dir = ds_dir
        self.sfm_model_dir = f"{ds_dir}/sparse"
        self.intrinsics_dict = {
            "c0": pycolmap.Camera(
                model="OPENCV",
                width=1024,
                height=768,
                params=[
                    868.993378,
                    866.063001,
                    525.942323,
                    420.042529,
                    -0.399431,
                    0.188924,
                    0.000153,
                    0.000571,
                ],
            ),
            "c1": pycolmap.Camera(
                model="OPENCV",
                width=1024,
                height=768,
                params=[
                    873.382641,
                    876.489513,
                    529.324138,
                    397.272397,
                    -0.397066,
                    0.181925,
                    0.000176,
                    -0.000579,
                ],
            ),
        }
        if train:
            self.images_dir_str = f"{self.ds_dir}/database"
            self.images_dir = Path(self.images_dir_str)
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
                self.image_name2id[image.name] = image_id
            self.image_id2points = {}
            self.pid2images = {}

            for img_id in tqdm(self.recon_images):
                pid_arr = self.recon_images[img_id].point3D_ids
                pid_arr = pid_arr[pid_arr >= 0]
                xyz_arr = np.zeros((pid_arr.shape[0], 3))
                for idx, pid in enumerate(pid_arr):
                    xyz_arr[idx] = self.recon_points[pid].xyz
                self.image_id2points[img_id] = xyz_arr
            self.img_ids = list(self.image_name2id.values())
        else:
            self.images_dir_str = f"{self.ds_dir}/query"
            self.images_dir = Path(self.images_dir_str)

            self.img_ids = [str(file) for file in self.images_dir.iterdir() if file.is_file()]

        self.train = train

    def _clear(self):
        if self.train:
            self.recon_images.clear()
            self.recon_cameras.clear()
            self.recon_points.clear()

    def _load_image(self, img_id):
        if self.train:
            name = self.recon_images[img_id].name
            name2 = str(self.images_dir / name)
        else:
            name2 = img_id
        image = io.imread(name2)

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image, name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]

            image, image_name = self._load_image(img_id)
            camera_id = self.recon_images[img_id].camera_id
            camera = self.recon_cameras[camera_id]
            camera = pycolmap.Camera(
                model=camera.model,
                width=int(camera.width),
                height=int(camera.height),
                params=camera.params,
            )
            qvec = self.recon_images[img_id].qvec
            tvec = self.recon_images[img_id].tvec
            pose_inv = dd_utils.return_pose_mat_no_inv(qvec, tvec)

            xyz_gt = self.image_id2points[img_id]
            pid_list = self.recon_images[img_id].point3D_ids
            mask = pid_list >= 0
            pid_list = pid_list[mask]
            uv_gt = self.recon_images[img_id].xys[mask]

            pose_inv = torch.from_numpy(pose_inv)

        else:
            img_id = self.img_ids[idx]
            image, image_name = self._load_image(img_id)
            cam_id = image_name.split("/")[-1].split("_")[2]
            camera = self.intrinsics_dict[cam_id]

            image = None
            img_id = image_name.split("/")[-1]
            pid_list = []
            pose_inv = None
            xyz_gt = None
            uv_gt = None

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            None,
            camera,
            xyz_gt,
            uv_gt,
        )

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


if __name__ == "__main__":
    testset = CMUDataset(train=False)
    for t in testset:
        continue
