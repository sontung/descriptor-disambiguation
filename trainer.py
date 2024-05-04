import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import faiss
import h5py
import numpy as np
import poselib
import torch
from pykdtree.kdtree import KDTree
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dd_utils
from ace_util import read_and_preprocess
from vae_test import VAEDataset
from dd_utils import concat_images_different_sizes


def retrieve_pid(pid_list, uv_gt, keypoints):
    tree = KDTree(keypoints.astype(uv_gt.dtype))
    dis, ind = tree.query(uv_gt)
    mask = dis < 5
    selected_pid = np.array(pid_list)[mask]
    return selected_pid, mask, ind


def compute_pose_error(pose, pose_gt):
    est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
    out_pose = torch.from_numpy(est_pose)

    # Calculate translation error.
    t_err = float(torch.norm(pose_gt[0:3, 3] - out_pose[0:3, 3]))

    gt_R = pose_gt[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()

    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return t_err, r_err


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
        using_global_descriptors,
        run_local_feature_detection_on_test_set=True,
        collect_code_book=True,
        lambda_val=1,
        normalize=False,
    ):
        self.feature_dim = feature_dim
        self.dataset = train_ds
        self.test_dataset = test_ds
        self.using_global_descriptors = using_global_descriptors
        self.global_feature_dim = global_feature_dim
        self.normalize = normalize

        self.name2uv = {}
        self.ds_name = self.dataset.ds_type
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.local_desc_model_name = local_desc_model.conf["name"]
        except AttributeError:
            if type(local_desc_model) == tuple:
                self.local_desc_model_name = "sdf2"
        self.global_desc_model_name = (
            f"{global_desc_model.conf['name']}_{global_feature_dim}"
        )
        self.local_desc_model = local_desc_model
        self.local_desc_conf = local_desc_conf

        self.global_desc_model = global_desc_model
        self.global_desc_conf = global_desc_conf

        if run_local_feature_detection_on_test_set:
            self.test_features_path = (
                f"output/{self.ds_name}/{self.local_desc_model_name}_features_test.h5"
            )
            if not os.path.isfile(self.test_features_path):
                features_h5 = h5py.File(
                    str(self.test_features_path), "a", libver="latest"
                )
                with torch.no_grad():
                    for example in tqdm(
                        self.test_dataset, desc="Detecting testing features"
                    ):
                        if example is None:
                            continue
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()
        else:
            self.test_features_path = None

        self.pca = None
        self.using_pca = False
        self.lambda_val = lambda_val
        print(f"using lambda val={self.lambda_val}")
        self.global_desc_mean = 0
        self.global_desc_std = 1

        self.local_features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )

        if self.using_global_descriptors:
            self.image2desc = self.collect_image_descriptors()
        else:
            self.image2desc = {}

        self.xyz_arr = None
        if collect_code_book:
            self.pid2descriptors = {}
            self.pid2count = {}
            self.improve_codebook()
            (
                self.pid2mean_desc,
                self.all_pid_in_train_set,
                self.pid2ind,
            ) = self.collect_descriptors()
            if self.pid2ind:
                self.all_ind_in_train_set = np.array(
                    [self.pid2ind[pid] for pid in self.all_pid_in_train_set]
                )
                self.ind2pid = {v: k for k, v in self.pid2ind.items()}
        else:
            self.pid2mean_desc = None
            self.all_pid_in_train_set = None
            self.pid2ind = None
            self.all_ind_in_train_set = None
            self.ind2pid = None

    def improve_codebook(self, vis=False):
        img_dir_str = self.dataset.images_dir_str
        available_images_dir = Path(img_dir_str)
        available_images = [
            str(file).split(f"{img_dir_str}/")[-1]
            for file in available_images_dir.rglob("*")
            if file.is_file()
        ]
        used_img_names = [
            self.dataset.recon_images[img_id].name for img_id in self.dataset.img_ids
        ]

        matches_h5 = h5py.File(
            str(
                f"outputs/aachen_v1.1/{self.local_desc_model_name}_nn.h5"
            ),
            "a",
            libver="latest",
        )
        features_h5 = h5py.File(
            str(f"outputs/aachen_v1.1/{self.local_desc_model_name}.h5"), "a", libver="latest"
        )
        features_db_h5 = h5py.File(self.local_features_path, "a", libver="latest")

        image2desc = {}
        if self.using_global_descriptors:
            for image_name in tqdm(available_images, desc="Processing global descriptors for extra images"):
                if image_name not in used_img_names:
                    image_name_for_matching_db = image_name.replace("/", "-")
                    if image_name_for_matching_db in matches_h5:
                        image_descriptor = self.produce_image_descriptor(
                            f"{img_dir_str}/{image_name}"
                        )
                        image2desc[image_name] = image_descriptor

        print(f"Got {len(image2desc)} extra images")
        count = 0
        for image_name in tqdm(image2desc, desc="Improving codebook with extra images"):
            image_name_for_matching_db = image_name.replace("/", "-")
            data = matches_h5[image_name_for_matching_db]
            for db_img in data:
                matches = data[db_img]
                indices = np.array(matches["matches0"])
                mask0 = indices > -1
                if np.sum(mask0) < 10:
                    continue
                uv0 = np.array(features_h5[image_name]["keypoints"])
                db_img_normal = db_img.replace("-", "/")
                uv1 = np.array(features_h5[db_img_normal]["keypoints"])
                descriptors0 = np.array(
                    features_h5[image_name]["descriptors"]
                ).T
                descriptors0 = descriptors0[mask0]
                uv0 = uv0[mask0]
                uv1 = uv1[indices[mask0]]

                db_img_id = self.dataset.image_name2id[db_img_normal]
                pid_list = self.dataset.image_id2pids[db_img_id]
                uv_gt = self.dataset.image_id2uvs[db_img_id]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv_gt, uv1)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                selected_descriptors = descriptors0[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = image2desc[image_name]
                    selected_descriptors = (
                        self.lambda_val * selected_descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors0.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in self.pid2descriptors:
                        self.pid2descriptors[pid] = selected_descriptors[idx]
                        self.pid2count[pid] = 1
                    else:
                        self.pid2count[pid] += 1
                        self.pid2descriptors[pid] = (
                            self.pid2descriptors[pid]
                            + selected_descriptors[idx]
                        )

                count += 1

                if vis:
                        img0 = cv2.imread(f"{img_dir_str}/{image_name}")
                        img1 = cv2.imread(f"{img_dir_str}/{db_img_normal}")
                        img2 = concat_images_different_sizes([img0, img1])
                        uv0 = uv0.astype(int)
                        uv1 = uv1.astype(int)
                        for idx in range(uv0.shape[0]):
                            u0, v0 = uv0[idx]
                            u1, v1 = uv1[idx]
                            cv2.circle(img2, (u0, v0), 10, (255, 0, 0, 255), -1)
                            cv2.circle(
                                img2,
                                (u1 + img0.shape[1], v1),
                                10,
                                (255, 0, 0, 255),
                                -1,
                            )
                            cv2.line(
                                img2,
                                (u0, v0),
                                (u1 + img0.shape[1], v1),
                                (255, 0, 0, 255),
                                2,
                            )
                        cv2.imwrite(f"debug/test{np.sum(mask)}-{count}.png", img2)

        matches_h5.close()
        features_h5.close()
        features_db_h5.close()
        print(f"Codebook improved from {count} pairs.")

    def improve_codebook2(self, vis=False):
        img_dir_str = self.dataset.images_dir_str
        available_images_dir = Path(img_dir_str)
        available_images = [
            str(file).split(f"{img_dir_str}/")[-1]
            for file in available_images_dir.rglob("*")
            if file.is_file()
        ]
        used_img_names = [
            self.dataset.recon_images[img_id].name for img_id in self.dataset.img_ids
        ]

        matches_h5 = h5py.File(
            str(
                "outputs/aachen_v1.1/r2d2_nn.h5"
            ),
            "a",
            libver="latest",
        )
        features_matches_h5 = h5py.File(
            str("outputs/aachen_v1.1/r2d2.h5"),
            "a",
            libver="latest",
        )
        features_h5 = h5py.File(
            str("outputs/aachen_v1.1/d2net.h5"), "a", libver="latest"
        )
        features_db_h5 = h5py.File(self.local_features_path, "a", libver="latest")

        image2desc = {}
        if self.using_global_descriptors:
            for image_name in tqdm(
                available_images, desc="Processing global descriptors for extra images"
            ):
                if image_name not in used_img_names:
                    image_name_for_matching_db = image_name.replace("/", "-")
                    if image_name_for_matching_db in matches_h5:
                        image_descriptor = self.produce_image_descriptor(
                            f"{img_dir_str}/{image_name}"
                        )
                        image2desc[image_name] = image_descriptor

        print(f"Got {len(image2desc)} extra images")
        count = 0
        for image_name in tqdm(
            image2desc, desc="Improving codebook with extra images"
        ):
            image_name_for_matching_db = image_name.replace("/", "-")
            data = matches_h5[image_name_for_matching_db]
            for db_img in data:
                matches = data[db_img]
                indices = np.array(matches["matches0"])
                mask0 = indices > -1
                if np.sum(mask0) < 10:
                    continue
                uv0 = np.array(features_matches_h5[image_name]["keypoints"])

                db_img_normal = db_img.replace("-", "/")
                uv1 = np.array(features_matches_h5[db_img_normal]["keypoints"])
                uv0 = uv0[mask0]
                uv1 = uv1[indices[mask0]]

                db_img_id = self.dataset.image_name2id[db_img_normal]
                pid_list = self.dataset.image_id2pids[db_img_id]
                uv_gt = self.dataset.image_id2uvs[db_img_id]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv_gt, uv1)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)
                selected_pid = selected_pid[ind2]
                uv0 = uv0[idx_arr]

                uv_matches = np.array(features_h5[image_name]["keypoints"])
                descriptors_matches = np.array(
                    features_h5[image_name]["descriptors"]
                ).T
                _, mask, ind = retrieve_pid(selected_pid, uv0, uv_matches)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                selected_descriptors = descriptors_matches[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = image2desc[image_name]
                    selected_descriptors = (
                        self.lambda_val * selected_descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors_matches.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in self.pid2descriptors:
                        self.pid2descriptors[pid] = selected_descriptors[idx]
                        self.pid2count[pid] = 1
                    else:
                        self.pid2count[pid] += 1
                        self.pid2descriptors[pid] = (
                            self.pid2descriptors[pid]
                            + selected_descriptors[idx]
                        )

                count += 1
        matches_h5.close()
        features_h5.close()
        features_db_h5.close()
        print(f"Codebook improved from {count} pairs.")

    def collect_image_descriptors(self, using_pca=False):
        file_name1 = f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
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
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    all_desc[idx] = image_descriptor
                    all_names.append(example[1])
                    idx += 1
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

        image2desc = {}
        if using_pca:
            self.pca = PCA(whiten=False, n_components=self.feature_dim)
            all_desc = self.pca.fit_transform(all_desc)
            self.using_pca = True

        self.global_desc_mean = np.mean(all_desc)
        self.global_desc_std = np.std(all_desc)

        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx, : self.feature_dim]
        return image2desc

    def produce_image_descriptor(self, name):
        with torch.no_grad():
            if (
                "mixvpr" in self.global_desc_model_name
                or "crica" in self.global_desc_model_name
                or "salad" in self.global_desc_model_name
                or "gcl" in self.global_desc_model_name
            ):
                image_descriptor = self.global_desc_model.process(name)
            else:
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
        if self.local_desc_model_name == "sdf2":
            model, extractor, conf = self.local_desc_model
            pred = extractor(
                model,
                img=torch.from_numpy(image).unsqueeze(0).cuda(),
                topK=conf["model"]["max_keypoints"],
                mask=None,
                conf_th=conf["model"]["conf_th"],
                scales=conf["model"]["scales"],
            )
            pred["descriptors"] = pred["descriptors"].T
        else:
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
            if self.using_pca:
                file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_pca.npy"
                file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_pca.npy"
                file_name3 = f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_pca.pkl"
            else:
                file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_{self.lambda_val}.npy"
                file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_{self.lambda_val}.npy"
                file_name3 = f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_{self.lambda_val}.pkl"
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

        if os.path.isfile(file_name1):
            print(f"Loading codebook from {file_name1}")
            pid2mean_desc = np.load(file_name1)
            all_pid = np.load(file_name2)
            afile = open(file_name3, "rb")
            pid2ind = pickle.load(afile)
            afile.close()
        else:
            if not os.path.isfile(self.local_features_path):
                features_h5 = h5py.File(
                    str(self.local_features_path), "a", libver="latest"
                )
                with torch.no_grad():
                    for example in tqdm(self.dataset, desc="Detecting features"):
                        if example is None:
                            continue
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()

            features_h5 = h5py.File(self.local_features_path, "r")

            for example in tqdm(self.dataset, desc="Collecting point descriptors"):
                if example is None:
                    continue
                try:
                    keypoints, descriptors = dd_utils.read_kp_and_desc(
                        example[1], features_h5
                    )
                except KeyError:
                    print(f"Cannot read {example[1]} from {self.local_features_path}")
                    sys.exit()
                pid_list = example[3]
                uv = example[-1]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                selected_descriptors = descriptors[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = self.image2desc[example[1]]
                    if self.normalize:
                        image_descriptor = (
                            image_descriptor - self.global_desc_mean
                        ) / self.global_desc_std
                    selected_descriptors = (
                        self.lambda_val * selected_descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in self.pid2descriptors:
                        self.pid2descriptors[pid] = selected_descriptors[idx]
                        self.pid2count[pid] = 1
                    else:
                        self.pid2count[pid] += 1
                        self.pid2descriptors[pid] = (
                            self.pid2descriptors[pid] + selected_descriptors[idx]
                        )

            features_h5.close()
            all_pid = list(self.pid2descriptors.keys())
            all_pid = np.array(all_pid)

            pid2mean_desc = np.zeros(
                (all_pid.shape[0], self.feature_dim),
                self.pid2descriptors[list(self.pid2descriptors.keys())[0]].dtype,
            )

            pid2ind = {}
            ind = 0
            for pid in self.pid2descriptors:
                pid2mean_desc[ind] = self.pid2descriptors[pid] / self.pid2count[pid]
                pid2ind[pid] = ind
                ind += 1
            if np.sum(np.isnan(pid2mean_desc)) > 0:
                print(f"NaN detected in codebook: {np.sum(np.isnan(pid2mean_desc))}")
            # np.save(file_name1, pid2mean_desc)
            # np.save(file_name2, all_pid)
            # with open(file_name3, "wb") as handle:
            #     pickle.dump(pid2ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_{self.lambda_val}.txt",
                "w",
            )
        else:
            result_file = open(
                f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}.txt",
                "w",
            )

        global_descriptors_path = f"output/{self.ds_name}/{self.global_desc_model_name}_{self.global_feature_dim}_desc_test.h5"
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
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
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                if self.using_global_descriptors:
                    image_descriptor = dd_utils.read_global_desc(
                        name, global_features_h5
                    )

                    if self.normalize:
                        image_descriptor = (
                            image_descriptor - self.global_desc_mean
                        ) / self.global_desc_std

                    descriptors = (
                        self.lambda_val * descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors.shape[1]]
                    )

                    # descriptors = 0.5 * (
                    #     descriptors + image_descriptor[: descriptors.shape[1]]
                    # )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                camera = example[6]

                camera_dict = {
                    "model": camera.model.name,
                    "height": camera.height,
                    "width": camera.width,
                    "params": camera.params,
                }
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera_dict,
                )

                qvec = " ".join(map(str, pose.q))
                tvec = " ".join(map(str, pose.t))

                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        features_h5.close()
        result_file.close()
        global_features_h5.close()

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
    def improve_codebook(self, vis=False):
        img_dir_str = self.dataset.images_dir_str
        available_images_dir = Path(img_dir_str)
        available_images = [
            str(file).split(f"{img_dir_str}/")[-1]
            for file in available_images_dir.rglob("*")
            if file.is_file()
        ]
        used_img_names = [
            self.dataset._process_id_to_name(img_id).split(f"{img_dir_str}/")[-1] for img_id in self.dataset.img_ids
        ]

        matches_h5 = h5py.File(
            str(
                f"/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/{self.local_desc_model_name}_nn.h5"
            ),
            "a",
            libver="latest",
        )
        features_h5 = h5py.File(
            str(f"/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/{self.local_desc_model_name}.h5"), "a", libver="latest"
        )
        features_db_h5 = h5py.File(self.local_features_path, "a", libver="latest")

        image2desc = {}
        if self.using_global_descriptors:
            for image_name in tqdm(available_images, desc="Processing global descriptors for extra images"):
                if image_name not in used_img_names:
                    image_name_for_matching_db = image_name.replace("/", "-")
                    if image_name_for_matching_db in matches_h5:
                        image_descriptor = self.produce_image_descriptor(
                            f"{img_dir_str}/{image_name}"
                        )
                        image2desc[image_name] = image_descriptor

        print(f"Got {len(image2desc)} extra images")
        count = 0
        for image_name in tqdm(image2desc, desc="Improving codebook with extra images"):
            image_name_for_matching_db = image_name.replace("/", "-")
            data = matches_h5[image_name_for_matching_db]
            for db_img in data:
                matches = data[db_img]
                indices = np.array(matches["matches0"])
                mask0 = indices > -1
                if np.sum(mask0) < 10:
                    continue
                uv0 = np.array(features_h5[image_name]["keypoints"])
                db_img_normal = db_img.replace("-", "/")
                uv1 = np.array(features_h5[db_img_normal]["keypoints"])
                descriptors0 = np.array(
                    features_h5[image_name]["descriptors"]
                ).T
                descriptors0 = descriptors0[mask0]
                uv0 = uv0[mask0]
                uv1 = uv1[indices[mask0]]

                db_img_id = self.dataset.image_name2id[db_img_normal]
                pid_list = self.dataset.image_id2pids[db_img_id]
                uv_gt = self.dataset.image_id2uvs[db_img_id]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv_gt, uv1)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                selected_descriptors = descriptors0[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = image2desc[image_name]
                    selected_descriptors = (
                        self.lambda_val * selected_descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors0.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in self.pid2descriptors:
                        self.pid2descriptors[pid] = selected_descriptors[idx]
                        self.pid2count[pid] = 1
                    else:
                        self.pid2count[pid] += 1
                        self.pid2descriptors[pid] = (
                            self.pid2descriptors[pid]
                            + selected_descriptors[idx]
                        )

                count += 1

        matches_h5.close()
        features_h5.close()
        features_db_h5.close()
        print(f"Codebook improved from {count} pairs.")

    def collect_descriptors(self, vis=False, reduce_map_size=True):
        if self.using_global_descriptors:
            if self.using_pca:
                file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_pca.npy"
                file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_pca.npy"
            else:
                file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
                file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}.npy"
        else:
            file_name1 = (
                f"output/{self.ds_name}/codebook_{self.local_desc_model_name}.npy"
            )
            file_name2 = (
                f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}.npy"
            )

        if reduce_map_size:
            index_map_file_name = f"output/{self.ds_name}/indices.npy"
            if os.path.isfile(index_map_file_name):
                inlier_ind = np.load(index_map_file_name)
            else:
                import open3d as o3d

                point_cloud = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(self.dataset.xyz_arr)
                )
                cl, inlier_ind = point_cloud.remove_radius_outlier(
                    nb_points=16, radius=5, print_progress=True
                )

                np.save(index_map_file_name, np.array(inlier_ind))

                # vis = o3d.visualization.Visualizer()
                # vis.create_window(width=1920, height=1025)
                # vis.add_geometry(point_cloud)
                # vis.run()
                # vis.destroy_window()
            img2points2 = {}
            inlier_ind_set = set(inlier_ind)
            for img in tqdm(
                self.dataset.image2points, desc="Removing outlier points in the map"
            ):
                pid_list = self.dataset.image2points[img]
                img2points2[img] = [pid for pid in pid_list if pid in inlier_ind_set]
                mask = [True if pid in inlier_ind_set else False for pid in pid_list]
                self.dataset.image2uvs[img] = np.array(self.dataset.image2uvs[img])[
                    mask
                ]
            self.dataset.image2points = img2points2

        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )
        if os.path.isfile(file_name1):
            pid2mean_desc = np.load(file_name1)
            all_pid = np.load(file_name2)
        else:
            if not os.path.isfile(features_path):
                features_h5 = h5py.File(str(features_path), "a", libver="latest")
                with torch.no_grad():
                    for example in tqdm(self.dataset, desc="Detecting features"):
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()

            pid2descriptors = {}
            pid2count = {}
            features_h5 = h5py.File(features_path, "r")

            for example in tqdm(self.dataset, desc="Collecting point descriptors"):
                keypoints, descriptors = dd_utils.read_kp_and_desc(
                    example[1], features_h5
                )
                pid_list = example[3]
                uv = example[-1]
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

                selected_descriptors = descriptors[idx_arr]
                if self.using_global_descriptors:
                    image_descriptor = self.image2desc[example[1]]
                    if self.normalize:
                        image_descriptor = (
                            image_descriptor - self.global_desc_mean
                        ) / self.global_desc_std
                    selected_descriptors = 0.5 * (
                        selected_descriptors + image_descriptor[: descriptors.shape[1]]
                    )

                for idx, pid in enumerate(selected_pid[ind2]):
                    if pid not in pid2descriptors:
                        pid2descriptors[pid] = selected_descriptors[idx]
                        pid2count[pid] = 1
                    else:
                        pid2count[pid] += 1
                        pid2descriptors[pid] = (
                            pid2descriptors[pid] + selected_descriptors[idx]
                        )

            features_h5.close()
            self.image2desc.clear()

            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            pid2mean_desc = np.zeros(
                (all_pid.shape[0], self.feature_dim),
                pid2descriptors[list(pid2descriptors.keys())[0]].dtype,
            )

            for ind, pid in enumerate(all_pid):
                pid2mean_desc[ind] = pid2descriptors[pid] / pid2count[pid]

            np.save(file_name1, pid2mean_desc)
            np.save(file_name2, all_pid)
        if pid2mean_desc.shape[0] > all_pid.shape[0]:
            pid2mean_desc = pid2mean_desc[all_pid]
        self.xyz_arr = self.dataset.xyz_arr[all_pid]
        return pid2mean_desc, all_pid, {}

    def legal_predict(
        self,
        uv_arr,
        features_ori,
        gpu_index_flat,
        remove_duplicate=False,
        return_pid=False,
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

        pred_scene_coords_b3 = self.xyz_arr[feature_indices]
        if return_pid:
            return uv_arr, pred_scene_coords_b3, feature_indices

        return uv_arr, pred_scene_coords_b3

    def evaluate(self):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)

        global_descriptors_path = f"output/{self.ds_name}/{self.global_desc_model_name}_{self.global_feature_dim}_desc_test.h5"
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
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

        if self.using_global_descriptors:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}.txt",
                "w",
            )
        else:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}.txt",
                "w",
            )

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                if self.using_global_descriptors:
                    image_descriptor = dd_utils.read_global_desc(
                        name, global_features_h5
                    )
                    if self.using_pca:
                        image_descriptor = self.pca.transform(
                            image_descriptor.reshape(1, -1)
                        )

                    if self.normalize:
                        image_descriptor = (
                            image_descriptor - self.global_desc_mean
                        ) / self.global_desc_std
                    descriptors = 0.5 * (
                        descriptors + image_descriptor[: descriptors.shape[1]]
                    )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                camera = example[6]
                camera_dict = {
                    "model": camera.model.name,
                    "height": camera.height,
                    "width": camera.width,
                    "params": camera.params,
                }
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera_dict,
                )

                qvec = " ".join(map(str, pose.q))
                tvec = " ".join(map(str, pose.t))

                image_id = "/".join(example[2].split("/")[1:])
                print(f"{image_id} {qvec} {tvec}", file=result_file)
            result_file.close()
        features_h5.close()
        global_features_h5.close()


class CMUTrainer(BaseTrainer):
    def clear(self):
        del self.pid2mean_desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])

        global_descriptors_path = (
            f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        )
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Collecting global descriptors for test set"
                ):
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        features_h5 = h5py.File(self.test_features_path, "r")
        global_features_h5 = h5py.File(global_descriptors_path, "r")
        query_results = []
        print(f"Reading global descriptors from {global_descriptors_path}")
        print(f"Reading local descriptors from {self.test_features_path}")

        if self.using_global_descriptors:
            result_file_name = f"output/{self.ds_name}/CMU_eval_{self.local_desc_model_name}_{self.global_desc_model_name}.txt"
        else:
            result_file_name = (
                f"output/{self.ds_name}/CMU_eval_{self.local_desc_model_name}.txt"
            )

        computed_images = {}
        if os.path.isfile(result_file_name):
            with open(result_file_name) as file:
                lines = [line.rstrip() for line in file]
            if len(lines) == len(self.test_dataset):
                print(f"Found result file at {result_file_name}. Skipping")
                return lines
            else:
                computed_images = {line.split(" ")[0]: line for line in lines}

        result_file = open(result_file_name, "w")
        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                if example is None:
                    continue
                name = example[1]
                image_id = example[2].split("/")[-1]
                if image_id in computed_images:
                    line = computed_images[image_id]
                else:
                    keypoints, descriptors = dd_utils.read_kp_and_desc(
                        name, features_h5
                    )

                    if self.using_global_descriptors:
                        image_descriptor = dd_utils.read_global_desc(
                            name, global_features_h5
                        )
                        if self.normalize:
                            image_descriptor = (
                                image_descriptor - self.global_desc_mean
                            ) / self.global_desc_std

                        descriptors = 0.5 * (
                            descriptors + image_descriptor[: descriptors.shape[1]]
                        )

                    uv_arr, xyz_pred = self.legal_predict(
                        keypoints,
                        descriptors,
                        gpu_index_flat,
                    )

                    camera = example[6]

                    camera_dict = {
                        "model": "OPENCV",
                        "height": camera.height,
                        "width": camera.width,
                        "params": camera.params,
                    }
                    pose, info = poselib.estimate_absolute_pose(
                        uv_arr,
                        xyz_pred,
                        camera_dict,
                    )

                    qvec = " ".join(map(str, pose.q))
                    tvec = " ".join(map(str, pose.t))
                    line = f"{image_id} {qvec} {tvec}"
                query_results.append(line)
                print(line, file=result_file)
        features_h5.close()
        global_features_h5.close()
        result_file.close()
        return query_results


class SevenScenesTrainer(BaseTrainer):
    def clear(self):
        del self.pid2mean_desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])

        global_descriptors_path = (
            f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        )
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Collecting global descriptors for test set"
                ):
                    if example is None:
                        continue
                    image_descriptor = self.produce_image_descriptor(example[1])
                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        features_h5 = h5py.File(self.test_features_path, "r")
        global_features_h5 = h5py.File(global_descriptors_path, "r")
        print(f"Reading global descriptors from {global_descriptors_path}")
        print(f"Reading local descriptors from {self.test_features_path}")
        rErrs = []
        tErrs = []
        pct5 = 0
        total_frames = 0
        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                if example is None:
                    continue
                name = example[1]
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)

                if self.using_global_descriptors:
                    image_descriptor = dd_utils.read_global_desc(
                        name, global_features_h5
                    )
                    descriptors = 0.5 * (
                        descriptors + image_descriptor[: descriptors.shape[1]]
                    )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                camera = example[6]
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera,
                )

                t_err, r_err = compute_pose_error(pose, example[4])
                rErrs.append(r_err)
                tErrs.append(t_err * 100)
                total_frames += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1

        pct5 = pct5 / total_frames * 100
        tErrs.sort()
        rErrs.sort()
        median_idx = total_frames // 2
        median_rErr = rErrs[median_idx]
        median_tErr = tErrs[median_idx]
        features_h5.close()
        global_features_h5.close()
        return median_tErr, median_rErr, pct5


class CambridgeLandmarksTrainer(BaseTrainer):
    def collect_descriptors(self, vis=False):
        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )

        if not os.path.isfile(features_path):
            features_h5 = h5py.File(str(features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Detecting features"):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

        pid2descriptors = {}
        features_h5 = h5py.File(features_path, "r")
        for example in tqdm(self.dataset, desc="Collecting point descriptors"):
            keypoints, descriptors = dd_utils.read_kp_and_desc(example[1], features_h5)
            pid_list = example[3]
            uv = example[-1]
            selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
            idx_arr, ind2 = np.unique(ind[mask], return_index=True)

            selected_descriptors = descriptors[idx_arr]
            if self.using_global_descriptors:
                image_descriptor = self.image2desc[example[1]]
                if self.normalize:
                    image_descriptor = (
                        image_descriptor - self.global_desc_mean
                    ) / self.global_desc_std

                selected_descriptors = (
                    self.lambda_val * selected_descriptors
                    + (1 - self.lambda_val) * image_descriptor[: descriptors.shape[1]]
                )

            for idx, pid in enumerate(selected_pid[ind2]):
                if pid not in pid2descriptors:
                    pid2descriptors[pid] = selected_descriptors[idx]
                else:
                    pid2descriptors[pid] = 0.5 * (
                        pid2descriptors[pid] + selected_descriptors[idx]
                    )

        features_h5.close()
        self.image2desc.clear()

        all_pid = list(pid2descriptors.keys())
        all_pid = np.array(all_pid)
        pid2mean_desc = np.zeros(
            (all_pid.shape[0], self.feature_dim),
            pid2descriptors[list(pid2descriptors.keys())[0]].dtype,
        )

        for ind, pid in enumerate(all_pid):
            pid2mean_desc[ind] = pid2descriptors[pid]

        if pid2mean_desc.shape[0] > all_pid.shape[0]:
            pid2mean_desc = pid2mean_desc[all_pid]
        self.xyz_arr = self.dataset.xyz_arr[all_pid]
        return pid2mean_desc, all_pid, {}

    def legal_predict(
        self,
        uv_arr,
        features_ori,
        gpu_index_flat,
        remove_duplicate=False,
        return_pid=False,
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

        pred_scene_coords_b3 = self.xyz_arr[feature_indices]
        if return_pid:
            return uv_arr, pred_scene_coords_b3, feature_indices

        return uv_arr, pred_scene_coords_b3

    def evaluate(self, return_name2err=False):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)

        global_descriptors_path = (
            f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        )
        if not os.path.isfile(global_descriptors_path):
            global_features_h5 = h5py.File(
                str(global_descriptors_path), "a", libver="latest"
            )
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
        rErrs = []
        tErrs = []
        testset = self.test_dataset
        name2err = {}
        with torch.no_grad():
            for example in tqdm(testset, desc="Computing pose for test set"):
                name = "/".join(example[1].split("/")[-2:])
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                if self.using_global_descriptors:
                    image_descriptor = dd_utils.read_global_desc(
                        name, global_features_h5
                    )

                    if self.normalize:
                        image_descriptor = (
                            image_descriptor - self.global_desc_mean
                        ) / self.global_desc_std

                    descriptors = (
                        self.lambda_val * descriptors
                        + (1 - self.lambda_val)
                        * image_descriptor[: descriptors.shape[1]]
                    )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                camera = example[6]
                pose, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera,
                )

                t_err, r_err = compute_pose_error(pose, example[4])
                name2err[name] = t_err

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

        features_h5.close()
        global_features_h5.close()
        total_frames = len(rErrs)
        assert total_frames == len(testset)

        # Compute median errors.
        tErrs.sort()
        rErrs.sort()
        median_idx = total_frames // 2
        median_rErr = rErrs[median_idx]
        median_tErr = tErrs[median_idx]
        if return_name2err:
            return median_tErr, median_rErr, name2err
        return median_tErr, median_rErr
