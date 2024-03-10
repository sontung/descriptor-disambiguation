import argparse
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import faiss
import h5py
import numpy as np
import pycolmap
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from pykdtree.kdtree import KDTree
from tqdm import tqdm

from ace_util import _strtobool
from ace_util import localize_pose_lib
from ace_util import read_and_preprocess
from dataset import AachenDataset
from scipy.spatial.transform import Rotation as Rotation

_logger = logging.getLogger(__name__)
# sys.path.append("../CosPlace")
# from cosplace_utils import load_image as load_image_cosplace
# from cosplace_utils import load_model as load_model_cosplace

# sys.path.append("../MixVPR")
# from mix_vpr_main import VPRModel
# from mix_vpr_demo import load_image as load_image_mix_vpr


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


def read_kp_and_desc(name, features_h5):
    pred = {}
    grp = features_h5[name]
    for k, v in grp.items():
        pred[k] = v

    pred = {k: np.array(v) for k, v in pred.items()}
    scale = pred["scale"]
    keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
    descriptors = pred["descriptors"].T
    return keypoints, descriptors


class TrainerACE:
    def __init__(self):
        self.dataset = AachenDataset()

        self.name2uv = {}
        self.ds_name = "aachen"
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        conf = {
            "r2d2": {
                "output": "feats-r2d2-n5000-r1024",
                "model": {"name": "r2d2", "max_keypoints": 5000,},
                "preprocessing": {"grayscale": False, "resize_max": 1024,},
            },
            "d2net-ss": {
                "output": "feats-d2net-ss",
                "model": {"name": "d2net", "multiscale": False,},
                "preprocessing": {"grayscale": False, "resize_max": 1600,},
            },
            "disk": {
                "output": "feats-disk",
                "model": {"name": "disk", "max_keypoints": 5000,},
                "preprocessing": {"grayscale": False, "resize_max": 1600,},
            },
            "netvlad": {
                "output": "global-feats-netvlad",
                "model": {"name": "netvlad"},
                "preprocessing": {"resize_max": 1024},
            },
            "openibl": {
                "output": "global-feats-openibl",
                "model": {"name": "openibl"},
                "preprocessing": {"resize_max": 1024},
            },
            "eigenplaces": {
                "output": "global-feats-eigenplaces",
                "model": {"name": "eigenplaces"},
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
        model_dict = conf["d2net-ss"]["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        self.encoder = Model(model_dict).eval().to(device)

        conf_ns = SimpleNamespace(**{**default_conf, **conf})
        conf_ns.grayscale = conf["d2net-ss"]["preprocessing"]["grayscale"]
        conf_ns.resize_max = conf["d2net-ss"]["preprocessing"]["resize_max"]
        self.conf = conf_ns

        # self.encoder_global = load_model_cosplace(
        #     "../CosPlace/models/resnet50_128.pth", "ResNet50"
        # )

        model_dict = conf["netvlad"]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        self.encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf["netvlad"]["preprocessing"]["resize_max"]
        self.conf_retrieval = conf_ns_retrieval

        # self.encoder_global = VPRModel(
        #     backbone_arch="resnet50",
        #     layers_to_crop=[4],
        #     agg_arch="MixVPR",
        #     agg_config={
        #         "in_channels": 1024,
        #         "in_h": 20,
        #         "in_w": 20,
        #         "out_channels": 64,
        #         "mix_depth": 4,
        #         "mlp_ratio": 1,
        #         "out_rows": 2,
        #     },
        # ).cuda()
        #
        # state_dict = torch.load(
        #     "../MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt"
        # )
        # self.encoder_global.load_state_dict(state_dict)
        # self.encoder_global.eval()

        self.image2desc = self.collect_image_descriptors()
        (
            self.pid2mean_desc,
            self.all_pid_in_train_set,
            self.pid2ind,
        ) = self.collect_descriptors()
        self.all_ind_in_train_set = np.array(
            [self.pid2ind[pid] for pid in self.all_pid_in_train_set]
        )
        self.ind2pid = {v: k for k, v in self.pid2ind.items()}

    def collect_image_descriptors(self):
        file_name1 = f"output/{self.ds_name}/image_desc_netvlad.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name_netvlad.npy"
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            all_desc = np.zeros((len(self.dataset), 512))
            all_names = []
            idx = 0
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collecting image descriptors"):
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = image_descriptor[:512]
                    all_names.append(example[1])
                    idx += 1
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        image2desc = {}
        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx]
        return image2desc

    def produce_image_descriptor(self, name):
        # image = load_image_cosplace(name, resize_test_imgs=True)
        # image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
        # image_descriptor = image_descriptor.squeeze().cpu().numpy()

        image, _ = read_and_preprocess(name, self.conf_retrieval)
        image_descriptor = (
            self.encoder_global({"image": torch.from_numpy(image).unsqueeze(0).cuda()})[
                "global_descriptor"
            ]
            .squeeze()
            .cpu()
            .numpy()
        )
        return image_descriptor

    def produce_local_descriptors(self, name, fd):
        image, scale = read_and_preprocess(name, self.conf)
        pred = self.encoder({"image": torch.from_numpy(image).unsqueeze(0).cuda()})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        dict_ = {
            "scale": scale,
            "keypoints": pred["keypoints"],
            "descriptors": pred["descriptors"],
        }

        try:
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            for k, v in dict_.items():
                grp.create_dataset(k, data=v)
        except OSError as error:
            if "No space left on device" in error.args[0]:
                print("No space left")
                del grp, fd[name]
            raise error

    def collect_descriptors(self, vis=False):
        file_name1 = f"output/{self.ds_name}/codebook_d2.npy"
        file_name2 = f"output/{self.ds_name}/all_pids_d2.npy"
        file_name3 = f"output/{self.ds_name}/pid2ind.pkl"
        features_path = f"output/{self.ds_name}/d2_features_train.h5"
        if os.path.isfile(file_name1):
            pid2mean_desc = np.load(file_name1)
            all_pid = np.load(file_name2)
            afile = open(file_name3, "rb")
            pid2ind = pickle.load(afile)
            afile.close()
        else:
            if not os.path.isfile(features_path):
                features_h5 = h5py.File(str(features_path), "a", libver="latest")
                with torch.no_grad():
                    for example in tqdm(self.dataset, desc="Detecting features"):
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()

            pid2descriptors = {}
            features_h5 = h5py.File(features_path, "r")
            for example in tqdm(self.dataset, desc="Collecting point descriptors"):
                keypoints, descriptors = read_kp_and_desc(example[1], features_h5)
                pid_list = example[3]
                uv = example[-1] + 0.5
                selected_pid, mask, ind = self.retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                if vis:
                    image = cv2.imread(example[1])
                    for u, v in uv.astype(int):
                        cv2.circle(image, (u, v), 5, (255, 0, 0))
                    for u, v in keypoints.astype(int):
                        cv2.circle(image, (u, v), 5, (0, 255, 0))
                    cv2.imwrite(f"debug/test{ind}.png", image)

                image_descriptor = self.image2desc[example[1]]
                # image_descriptor = np.mean(descriptors, 0)

                selected_descriptors = descriptors[idx_arr]
                selected_descriptors = 0.5 * (
                    selected_descriptors + image_descriptor[: descriptors.shape[1]]
                )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in pid2descriptors:
                        pid2descriptors[pid] = selected_descriptors[idx]
                    else:
                        pid2descriptors[pid] = 0.5*(selected_descriptors[idx]+pid2descriptors[pid])

            features_h5.close()
            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            desc_dim = pid2descriptors[list(pid2descriptors.keys())[0]][0].shape[0]
            pid2mean_desc = np.zeros(
                (len(self.dataset.recon_points), desc_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]][0].dtype,
            )

            pid2ind = {}
            ind = 0
            for pid in pid2descriptors:
                pid2mean_desc[ind] = pid2descriptors[pid]
                pid2ind[pid] = ind
                ind += 1
            np.save(file_name1, pid2mean_desc)
            np.save(file_name2, all_pid)
            with open(file_name3, "wb") as handle:
                pickle.dump(pid2ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return pid2mean_desc, all_pid, pid2ind

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """
        test_set = AachenDataset(train=False)

        features_path = f"output/{self.ds_name}/d2_features_test.h5"
        if not os.path.isfile(features_path):
            features_h5 = h5py.File(str(features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(test_set, desc="Detecting testing features"):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

        index = faiss.IndexFlatL2(512)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])
        result_file = open(f"output/{self.ds_name}/Aachen_v1_1_eval_dd.txt", "w")
        features_h5 = h5py.File(features_path, "r")

        with torch.no_grad():
            for example in tqdm(test_set, desc="Computing pose for test set"):
                keypoints, descriptors = read_kp_and_desc(example[1], features_h5)
                image_descriptor = self.produce_image_descriptor(example[1])

                # image_descriptor = np.mean(descriptors, 0)

                # image = load_image_mix_vpr(image_name)
                # image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                # image_descriptor = image_descriptor.squeeze().cpu().numpy()
                descriptors = 0.5 * (
                    descriptors + image_descriptor[: descriptors.shape[1]]
                )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints, descriptors, gpu_index_flat,
                )

                camera = example[6]
                res = pycolmap.absolute_pose_estimation(uv_arr, xyz_pred, camera)
                mat = res["cam_from_world"]
                qvec = " ".join(map(str, mat.rotation.quat[[3, 0, 1, 2]]))
                tvec = " ".join(map(str, mat.translation))
                image_id = example[2].split("/")[-1]
                print(
                    f"{image_id} {qvec} {tvec}", file=result_file
                )
        features_h5.close()

    def retrieve_pid(self, pid_list, uv_gt, keypoints):
        tree = KDTree(keypoints.astype(uv_gt.dtype))
        dis, ind = tree.query(uv_gt)
        mask = dis < 5
        selected_pid = np.array(pid_list)[mask]
        return selected_pid, mask, ind

    def legal_predict(
        self, uv_arr, features_ori, gpu_index_flat, remove_duplicate=False
    ):
        distances, feature_indices = gpu_index_flat.search(features_ori, 1)

        feature_indices = feature_indices.ravel()

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

        pid_pred = [
            self.ind2pid[ind] for ind in self.all_ind_in_train_set[feature_indices]
        ]
        pred_scene_coords_b3 = np.array(
            [self.dataset.recon_points[pid].xyz for pid in pid_pred]
        )

        return uv_arr, pred_scene_coords_b3


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = TrainerACE()
    # trainer.evaluate_with_hloc()
    trainer.evaluate()
    # trainer.test_model()
    # trainer.train()
