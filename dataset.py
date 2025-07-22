import logging
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torchvision.transforms
import ace_util
import colmap_read
import dd_utils
from PIL import Image

_logger = logging.getLogger(__name__)


CONDITIONS = [
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
]


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


def read_train_poses(a_file, cl=False):
    with open(a_file) as file:
        lines = [line.rstrip() for line in file]
    if cl:
        lines = lines[4:]
    name2mat = {}
    for line in lines:
        img_name, *matrix = line.split(" ")
        if len(matrix) == 16:
            matrix = np.array(matrix, float).reshape(4, 4)
        name2mat[img_name] = matrix
    return name2mat


class CambridgeLandmarksDataset(Dataset):
    def __init__(self, root_dir, ds_name, train=True):
        self.using_sfm_poses = True
        self.image_name2id = None
        self.train = train
        self.ds_type = ds_name

        # Setup data paths.
        sift_model_name = "CambridgeLandmarks_Colmap_Retriangulated_1024px"

        if self.train:
            self.sfm_model_dir = f"{str(root_dir)}/{ds_name}/sfm_sift_scaled"
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
            self.names = read_train_poses(
                f"{root_dir}/{sift_model_name}/{ds_name}/list_db.txt"
            )
            self.pid2images = {}

            Path(f"output/{self.ds_type}").mkdir(parents=True, exist_ok=True)

            xyz_arr = np.zeros((len(self.recon_points), 3))
            all_pids = np.zeros(len(self.recon_points), dtype=int)
            for idx, pid in enumerate(list(self.recon_points.keys())):
                xyz_arr[idx] = self.recon_points[pid].xyz
                all_pids[idx] = pid

            self.image_id2pids = {}
            self.image_id2uvs = {}
            for img_id in tqdm(self.recon_images, desc="Gathering points per image"):
                pid_arr = self.recon_images[img_id].point3D_ids
                mask = pid_arr >= 0
                self.image_id2pids[img_id] = pid_arr[mask]
                self.image_id2uvs[img_id] = self.recon_images[img_id].xys[mask]
            self.img_ids = list(self.image_name2id.values())
            self.img_names = list(self.names.keys())

        else:
            self.sfm_model_dir = f"{root_dir}/{sift_model_name}/{ds_name}/empty_all"
            self.recon_images = colmap_read.read_images_text(
                f"{self.sfm_model_dir}/images.txt"
            )
            self.recon_cameras = colmap_read.read_cameras_text(
                f"{self.sfm_model_dir}/cameras.txt"
            )
            self.recon_points = colmap_read.read_points3D_text(
                f"{self.sfm_model_dir}/points3D.txt"
            )
            self.image_name2id = {}
            for image_id, image in self.recon_images.items():
                self.image_name2id[image.name] = image_id
            self.name2params = read_intrinsic(
                f"{root_dir}/{ds_name}/query_list_with_intrinsics.txt"
            )
            self.img_names = list(self.name2params.keys())
        self.root_dir = root_dir
        self.images_dir = f"{root_dir}/{ds_name}"

        conf, default_conf = dd_utils.hloc_conf_for_all_models()
        self.conf_ns = SimpleNamespace(**{**default_conf, **conf})
        self.conf_ns.grayscale = True
        self.conf_ns.resize_max = 1024

    def _load_image(self, img_id):
        name = self.recon_images[img_id].name
        name2 = f"{self.images_dir}/{name}"
        image, scale = ace_util.read_and_preprocess(name2, self.conf_ns)

        return image[0], name2, scale

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]

            image, image_name, scale = self._load_image(img_id)
            assert image.shape == (576, 1024)
            camera_id = self.recon_images[img_id].camera_id
            camera = self.recon_cameras[camera_id]
            focal, cx, cy, k = camera.params
            intrinsics = torch.eye(3)

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy
            pose_inv = self.recon_images[img_id]

            pid_list = self.image_id2pids[img_id]
            uv_gt = self.image_id2uvs[img_id] / scale

            # if len(pid_list) > 0:
            #     qvec = self.recon_images[img_id].qvec
            #     tvec = self.recon_images[img_id].tvec
            #     pose_mat = dd_utils.return_pose_mat_no_inv(qvec, tvec)
            #     pose_mat = torch.from_numpy(pose_mat)
            #     uv_gt2 = project_using_pose(
            #         pose_mat.unsqueeze(0).cuda().float(),
            #         intrinsics.unsqueeze(0).cuda().float(),
            #         np.array([self.recon_points[pid].xyz for pid in pid_list]),
            #     )
            #     mask = np.mean(np.abs(uv_gt-uv_gt2), 1) < 5
            #     pid_list = pid_list[mask]
            #     uv_gt = uv_gt[mask]

            xyz_gt = None

            # import cv2
            # image = cv2.imread(image_name)
            # uv_gt2 = uv_gt/scale
            # for x, y in uv_gt2.astype(int):
            #     cv2.circle(image, (x, y), 5, (255, 0, 0))
            # cv2.imwrite(f"debug/test.png", image)

            # pose_inv = torch.from_numpy(pose_inv)

        else:
            name1 = self.img_names[idx]
            image_name = f"{self.images_dir}/{name1}"
            img_id = self.image_name2id[name1]
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
            pid_list = []
            # qvec = self.recon_images[img_id].qvec
            # tvec = self.recon_images[img_id].tvec
            # pose_inv = dd_utils.return_pose_mat_no_inv(qvec, tvec)
            pose_inv = self.recon_images[img_id]
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

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx)


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

            Path(f"output/{self.ds_type}").mkdir(parents=True, exist_ok=True)

            xyz_arr = np.zeros((len(self.recon_points), 3))
            all_pids = np.zeros(len(self.recon_points), dtype=int)
            for idx, pid in enumerate(list(self.recon_points.keys())):
                xyz_arr[idx] = self.recon_points[pid].xyz
                all_pids[idx] = pid

            self.image_id2pids = {}
            self.image_id2uvs = {}
            for img_id in tqdm(self.recon_images, desc="Gathering points per image"):
                pid_arr = self.recon_images[img_id].point3D_ids
                mask = pid_arr >= 0
                self.image_id2pids[img_id] = pid_arr[mask]
                self.image_id2uvs[img_id] = self.recon_images[img_id].xys[mask]
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

        return None, name2

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
            pose_inv = dd_utils.return_pose_mat_no_inv(qvec, tvec)
            pose_inv = torch.from_numpy(pose_inv)

            pid_list = self.image_id2pids[img_id]
            uv_gt = self.image_id2uvs[img_id]

            xyz_gt = None

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


class RobotCarDataset(Dataset):
    images_dir_str: str

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
                self.rgb_arr,
            ) = ace_util.read_nvm_file(self.sfm_model_dir)
            self.name2image = {v: k for k, v in self.image2name.items()}
            self.img_ids = list(self.image2name.keys())

            self.name2mat = read_train_poses(self.test_file1)

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
                self.name2mat = read_train_poses(self.test_file1)
            else:
                self.name2mat = read_train_poses(self.test_file2)
            self.img_ids = list(self.name2mat.keys())

        return

    def _process_id_to_name(self, img_id):
        name = self.image2name[img_id].split("./")[-1]
        name2 = str(self.images_dir / name).replace(".png", ".jpg")
        return name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        if self.train:
            img_id = self.img_ids[idx]
            image_name = self._process_id_to_name(img_id)
            if type(self.image2pose[img_id]) == list:
                qw, qx, qy, qz, tx, ty, tz = self.image2pose[img_id]
                tx, ty, tz = -(
                    Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    @ np.array([tx, ty, tz])
                )
                pose_mat = dd_utils.return_pose_mat_no_inv(
                    [qw, qx, qy, qz], [tx, ty, tz]
                )
            else:
                pose_mat = self.image2pose[img_id]
            image = None
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

            intrinsics[0, 0] = focal
            intrinsics[1, 1] = focal
            intrinsics[0, 2] = cx
            intrinsics[1, 2] = cy

            pid_list = self.image2points[img_id]
            xyz_gt = self.xyz_arr[pid_list]

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
                pose_inv = torch.from_numpy(self.name2mat[name0])
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

            for img_id in self.recon_images:
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

            self.img_ids = [
                str(file) for file in self.images_dir.iterdir() if file.is_file()
            ]

        self.train = train

    def clear(self):
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
        try:
            image = cv2.imread(name2)
        except ValueError or FileNotFoundError:
            return None, name2

        return image, name2

    def __len__(self):
        return len(self.img_ids)

    def _get_single_item(self, idx):
        img_id = self.img_ids[idx]
        image, image_name = self._load_image(img_id)
        if image is None:
            print(f"Warning: cannot read image at {image_name}")
            return None

        if self.train:
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

class CricaInferenceDataset(Dataset):
    def __init__(self, image_names):
        self.image_names = image_names
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image = torchvision.transforms.functional.resize(image, (224, 224))

        return image


if __name__ == "__main__":
    # testset = CambridgeLandmarksDataset(
    #     train=True, ds_name="GC", root_dir="../ace/datasets/Cambridge_GreatCourt"
    # )
    # for t in testset:
    #     continue

    # g = AachenDataset(train=False)
    # g[0]
    val_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=True)
    print()
