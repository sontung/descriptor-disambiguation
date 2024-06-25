import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pycolmap
import torch
from hloc.pipelines.RobotCar.pipeline import CONDITIONS
from pykdtree.kdtree import KDTree
from scipy.spatial.transform import Rotation
from skimage import color
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torchvision.transforms
import ace_util
import colmap_read
import dd_utils
from ace_util import project_using_pose
from PIL import Image

_logger = logging.getLogger(__name__)


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

            index_map_file_name = f"output/{self.ds_type}/indices.npy"
            Path(f"output/{self.ds_type}").mkdir(parents=True, exist_ok=True)

            xyz_arr = np.zeros((len(self.recon_points), 3))
            all_pids = np.zeros(len(self.recon_points), dtype=int)
            for idx, pid in enumerate(list(self.recon_points.keys())):
                xyz_arr[idx] = self.recon_points[pid].xyz
                all_pids[idx] = pid

            if os.path.isfile(index_map_file_name):
                inlier_ind = np.load(index_map_file_name)
            else:
                import open3d as o3d

                point_cloud = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(xyz_arr)
                )
                cl, inlier_ind = point_cloud.remove_radius_outlier(
                    nb_points=16, radius=5, print_progress=True
                )

                np.save(index_map_file_name, np.array(inlier_ind))

                # vis = o3d.visualization.Visualizer()
                # vis.create_window(width=1920, height=1025)
                # vis.add_geometry(cl)
                # vis.run()
                # vis.destroy_window()

            self.good_pids = set(all_pids[inlier_ind])
            # self.good_pids = set(all_pids)
            self.image_id2pids = {}
            self.image_id2uvs = {}
            for img_id in tqdm(self.recon_images, desc="Gathering points per image"):
                pid_arr = self.recon_images[img_id].point3D_ids
                # mask = [True if pid in self.good_pids else False for pid in pid_arr]
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
                self.name2mat = read_train_poses(self.test_file1)
            else:
                self.name2mat = read_train_poses(self.test_file2)
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
                tx, ty, tz = -(
                    Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    @ np.array([tx, ty, tz])
                )
                pose_mat = dd_utils.return_pose_mat_no_inv(
                    [qw, qx, qy, qz], [tx, ty, tz]
                )
            else:
                pose_mat = self.image2pose[img_id]

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
            image = io.imread(name2)
        except ValueError:
            return None, name2

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

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


class SevenScenesDataset(Dataset):
    def __init__(self, ds_name, img_dir, sfm_model_dir, train=True):
        self.ds_type = f"7scenes_{ds_name}"
        self.img_dir = img_dir
        self.sfm_model_dir = sfm_model_dir

        self.test_file = f"{self.sfm_model_dir}/list_test.txt"
        with open(self.test_file) as file:
            self.test_images = [line.rstrip() for line in file]

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

        if train:
            self.img_dir = f"{self.img_dir}/train/rgb"
            self.img_ids = [
                img_name
                for img_name in self.image_name2id
                if img_name not in self.test_images
            ]
        else:
            self.img_dir = f"{self.img_dir}/test/rgb"
            self.img_ids = self.test_images

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image_name = f"{self.img_dir}/{img_id.replace('/', '-')}"
        image = io.imread(image_name)
        sfm_image_id = self.image_name2id[img_id]

        camera_id = self.recon_images[sfm_image_id].camera_id
        camera = self.recon_cameras[camera_id]
        camera_dict = {
            "model": camera.model,
            "height": int(camera.height),
            "width": int(camera.width),
            "params": camera.params,
        }
        qvec = self.recon_images[sfm_image_id].qvec
        tvec = self.recon_images[sfm_image_id].tvec
        pose_inv = dd_utils.return_pose_mat_no_inv(qvec, tvec)
        pose_inv = torch.from_numpy(pose_inv)
        xyz_gt = self.image_id2points[sfm_image_id]
        pid_list = self.recon_images[sfm_image_id].point3D_ids
        mask = pid_list >= 0
        pid_list = pid_list[mask]
        uv_gt = self.recon_images[sfm_image_id].xys[mask]

        return (
            image,
            image_name,
            img_id,
            pid_list,
            pose_inv,
            None,
            camera_dict,
            xyz_gt,
            uv_gt,
        )


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
    train_ds_ = CambridgeLandmarksDataset(
        train=True,
        ds_name="ShopFacade",
        root_dir=f"datasets/cambridge",
    )
    for _ in train_ds_:
        continue
