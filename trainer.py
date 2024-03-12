import os
import pickle
from pathlib import Path

import cv2
import faiss
import h5py
import numpy as np
import pycolmap
import torch
from pykdtree.kdtree import KDTree
from tqdm import tqdm

import dd_utils
from ace_util import read_and_preprocess


def retrieve_pid(pid_list, uv_gt, keypoints):
    tree = KDTree(keypoints.astype(uv_gt.dtype))
    dis, ind = tree.query(uv_gt)
    mask = dis < 5
    selected_pid = np.array(pid_list)[mask]
    return selected_pid, mask, ind


class BaseTrainer:
    def __init__(
        self,
        train_ds,
        test_ds,
        feature_dim,
        global_feature_dim,
        local_desc_model,
        global_desc_model,
        local_desc_conf,
        global_desc_conf,
        using_global_descriptors=True,
    ):
        self.feature_dim = feature_dim
        self.dataset = train_ds
        self.test_dataset = test_ds
        self.using_global_descriptors = using_global_descriptors
        self.global_feature_dim = global_feature_dim

        self.name2uv = {}
        self.ds_name = self.dataset.ds_type
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.local_desc_model_name = local_desc_model.conf["name"]
        self.global_desc_model_name = global_desc_model.conf["name"]
        self.local_desc_model = local_desc_model
        self.local_desc_conf = local_desc_conf

        self.global_desc_model = global_desc_model
        self.global_desc_conf = global_desc_conf

        self.test_features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_test.h5"
        )
        if not os.path.isfile(self.test_features_path):
            features_h5 = h5py.File(str(self.test_features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Detecting testing features"
                ):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

        if self.using_global_descriptors:
            self.image2desc = self.collect_image_descriptors()
        else:
            self.image2desc = {}
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
        file_name1 = (
            f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}.npy"
        )
        file_name2 = (
            f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}.npy"
        )
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            all_desc = np.zeros((len(self.dataset), self.global_feature_dim))
            all_names = []
            idx = 0
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collecting image descriptors"):
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = image_descriptor
                    all_names.append(example[1])
                    idx += 1
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        image2desc = {}
        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx, :self.feature_dim]
        return image2desc

    def produce_image_descriptor(self, name):
        image, _ = read_and_preprocess(name, self.global_desc_conf)
        image_descriptor = (
            self.global_desc_model(
                {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
            )["global_descriptor"]
            .squeeze()
            .cpu()
            .numpy()
        )
        return image_descriptor

    def produce_local_descriptors(self, name, fd):
        image, scale = read_and_preprocess(name, self.local_desc_conf)
        # print(name)
        # print(image.shape, name)
        pred = self.local_desc_model(
            {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
        )
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        dict_ = {
            "scale": scale,
            "keypoints": pred["keypoints"],
            "descriptors": pred["descriptors"],
        }
        dd_utils.write_to_h5_file(fd, name, dict_)

    def collect_descriptors(self, vis=False):
        if self.using_global_descriptors:
            file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name3 = f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}_{self.global_desc_model_name}.pkl"
        else:
            file_name1 = (
                f"output/{self.ds_name}/codebook_{self.local_desc_model_name}.npy"
            )
            file_name2 = (
                f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}.npy"
            )
            file_name3 = (
                f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}.pkl"
            )

        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
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
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                # idx_arr, ind2 = np.unique(ind[mask], return_index=True)
                idx_arr = ind[mask]

                if vis:
                    image = cv2.imread(example[1])
                    for u, v in uv.astype(int):
                        cv2.circle(image, (u, v), 5, (255, 0, 0))
                    for u, v in keypoints.astype(int):
                        cv2.circle(image, (u, v), 5, (0, 255, 0))
                    cv2.imwrite(f"debug/test{ind}.png", image)

                selected_descriptors = descriptors[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = self.image2desc[example[1]]
                    selected_descriptors = 0.5 * (
                        selected_descriptors + image_descriptor[: descriptors.shape[1]]
                    )

                # for idx, pid in enumerate(selected_pid[ind2]):
                for idx, pid in enumerate(selected_pid):
                    pid2descriptors.setdefault(pid, []).append(
                        selected_descriptors[idx]
                    )

            features_h5.close()
            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            pid2mean_desc = np.zeros(
                (len(self.dataset.recon_points), self.feature_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]][0].dtype,
            )

            pid2ind = {}
            ind = 0
            for pid in pid2descriptors:
                pid2mean_desc[ind] = np.mean(pid2descriptors[pid], 0)
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

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])
        if self.using_global_descriptors:
            result_file = open(
                f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}_{self.global_desc_model_name}.txt",
                "w",
            )
        else:
            result_file = open(
                f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}.txt",
                "w",
            )

        global_descriptors_path = (
            f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        )
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(str(global_descriptors_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Collecting global descriptors for test set"
                ):
                    image_descriptor = self.produce_image_descriptor(example[1])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        features_h5 = h5py.File(self.test_features_path, "r")
        global_features_h5 = h5py.File(global_descriptors_path, "r")

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                keypoints, descriptors = dd_utils.read_kp_and_desc(
                    name, features_h5
                )
                if self.using_global_descriptors:
                    image_descriptor = np.array(global_features_h5[name]["global_descriptor"])
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


class RobotCarTrainer(BaseTrainer):
    def __init__(self, train_ds, test_ds, feature_dim, global_feature_dim, local_desc_model, global_desc_model,
                 local_desc_conf, global_desc_conf, using_global_descriptors):
        super().__init__(train_ds, test_ds, feature_dim, global_feature_dim, local_desc_model, global_desc_model,
                         local_desc_conf, global_desc_conf, using_global_descriptors)

    def collect_descriptors(self, vis=False):
        if self.using_global_descriptors:
            file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name3 = f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}_{self.global_desc_model_name}.pkl"
        else:
            file_name1 = (
                f"output/{self.ds_name}/codebook_{self.local_desc_model_name}.npy"
            )
            file_name2 = (
                f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}.npy"
            )
            file_name3 = (
                f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}.pkl"
            )

        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
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
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                if vis:
                    image = cv2.imread(example[1])
                    for u, v in uv.astype(int):
                        cv2.circle(image, (u, v), 5, (255, 0, 0))
                    for u, v in keypoints.astype(int):
                        cv2.circle(image, (u, v), 5, (0, 255, 0))
                    cv2.imwrite(f"debug/test{ind}.png", image)

                selected_descriptors = descriptors[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = self.image2desc[example[1]]
                    selected_descriptors = 0.5 * (
                            selected_descriptors + image_descriptor[: descriptors.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    pid2descriptors.setdefault(pid, []).append(
                        selected_descriptors[idx]
                    )

            features_h5.close()
            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            pid2mean_desc = np.zeros(
                (len(self.dataset.recon_points), self.feature_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]][0].dtype,
            )

            pid2ind = {}
            ind = 0
            for pid in pid2descriptors:
                pid2mean_desc[ind] = np.mean(pid2descriptors[pid], 0)
                pid2ind[pid] = ind
                ind += 1
            np.save(file_name1, pid2mean_desc)
            np.save(file_name2, all_pid)
            with open(file_name3, "wb") as handle:
                pickle.dump(pid2ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return pid2mean_desc, all_pid, pid2ind
