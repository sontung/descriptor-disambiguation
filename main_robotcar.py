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

import dd_utils
from ace_util import _strtobool
from ace_util import localize_pose_lib
from ace_util import read_and_preprocess
from dataset import RobotCarDataset
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


class TrainerACE:
    def __init__(self):
        self.dataset = RobotCarDataset(train=True)

        self.name2uv = {}
        self.ds_name = "robotcar"
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = 128
        conf, default_conf = dd_utils.hloc_conf_for_all_models()
        self.local_desc_model = "r2d2"
        model_dict = conf[self.local_desc_model]["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        self.encoder = Model(model_dict).eval().to(device)

        conf_ns = SimpleNamespace(**{**default_conf, **conf})
        conf_ns.grayscale = conf[self.local_desc_model]["preprocessing"]["grayscale"]
        conf_ns.resize_max = conf[self.local_desc_model]["preprocessing"]["resize_max"]
        self.conf = conf_ns

        self.retrieval_model = "eigenplaces"
        model_dict = conf[self.retrieval_model]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        model_dict.update({'variant': 'EigenPlaces', 'backbone': 'ResNet101', 'fc_output_dim': 128})
        self.encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf[self.retrieval_model]["preprocessing"][
            "resize_max"
        ]
        self.conf_retrieval = conf_ns_retrieval

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
        file_name1 = f"output/{self.ds_name}/image_desc_{self.retrieval_model}_{self.feature_dim}.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.retrieval_model}_{self.feature_dim}.npy"
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            all_desc = np.zeros((len(self.dataset), self.feature_dim))
            all_names = []
            idx = 0
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collecting image descriptors"):
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = image_descriptor[: self.feature_dim]
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
        file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model}_{self.retrieval_model}.npy"
        file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model}_{self.retrieval_model}.npy"
        file_name3 = f"output/{self.ds_name}/pid2ind_{self.retrieval_model}.pkl"
        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model}_features_train.h5"
        )
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
                keypoints, descriptors = dd_utils.read_kp_and_desc(
                    example[1], features_h5
                )
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

                selected_descriptors = descriptors[idx_arr]
                selected_descriptors = 0.5 * (
                    selected_descriptors + image_descriptor[: descriptors.shape[1]]
                )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in pid2descriptors:
                        pid2descriptors[pid] = selected_descriptors[idx]
                    else:
                        pid2descriptors[pid] = 0.5 * (
                            selected_descriptors[idx] + pid2descriptors[pid]
                        )

            features_h5.close()
            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            desc_dim = pid2descriptors[list(pid2descriptors.keys())[0]].shape[0]
            pid2mean_desc = np.zeros(
                (len(self.dataset.recon_points), desc_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]].dtype,
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

        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model}_features_test.h5"
        )
        if not os.path.isfile(features_path):
            features_h5 = h5py.File(str(features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(test_set, desc="Detecting testing features"):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])
        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model}_{self.retrieval_model}.txt",
            "w",
        )
        features_h5 = h5py.File(features_path, "r")

        with torch.no_grad():
            for example in tqdm(test_set, desc="Computing pose for test set"):
                keypoints, descriptors = dd_utils.read_kp_and_desc(
                    example[1], features_h5
                )
                image_descriptor = self.produce_image_descriptor(example[1])

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
                print(f"{image_id} {qvec} {tvec}", file=result_file)
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
    trainer.evaluate()
