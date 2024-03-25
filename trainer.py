import math
import os
import pickle
import sys
from pathlib import Path
import poselib

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
        self.global_desc_model_name = (
            f"{global_desc_model.conf['name']}{global_feature_dim}"
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

        if self.using_global_descriptors:
            self.image2desc = self.collect_image_descriptors()
        else:
            self.image2desc = {}

        self.xyz_arr = None
        if collect_code_book:
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
        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx, : self.feature_dim]
        return image2desc

    def produce_image_descriptor(self, name):
        if "mixvpr" in self.global_desc_model_name:
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
                        if example is None:
                            continue
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()

            pid2descriptors = {}
            features_h5 = h5py.File(features_path, "r")
            for example in tqdm(self.dataset, desc="Collecting point descriptors"):
                if example is None:
                    continue
                try:
                    keypoints, descriptors = dd_utils.read_kp_and_desc(
                        example[1], features_h5
                    )
                except KeyError:
                    print(f"Cannot read {example[1]} from {features_path}")
                    sys.exit()
                pid_list = example[3]
                uv = example[-1] + 0.5
                selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
                idx_arr, ind2 = np.unique(ind[mask], return_index=True)

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
                    image_descriptor = np.array(
                        global_features_h5[name]["global_descriptor"]
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
                res = pycolmap.absolute_pose_estimation(uv_arr, xyz_pred, camera)
                mat = res["cam_from_world"]
                qvec = " ".join(map(str, mat.rotation.quat[[3, 0, 1, 2]]))
                tvec = " ".join(map(str, mat.translation))
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
    def collect_descriptors(self, vis=False):
        if self.using_global_descriptors:
            file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
        else:
            file_name1 = (
                f"output/{self.ds_name}/codebook_{self.local_desc_model_name}.npy"
            )
            file_name2 = (
                f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}.npy"
            )

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
                    selected_descriptors = 0.5 * (
                        selected_descriptors + image_descriptor[: descriptors.shape[1]]
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

    def test(self, test_dataset):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)
        # global_descriptors_path = (
        #     f"output/{self.ds_name}/{self.global_desc_model_name}_desc_test.h5"
        # )
        # global_features_h5 = h5py.File(global_descriptors_path, "r")

        features_path = f"output/{self.ds_name}/{self.local_desc_model_name}_features_train_small.h5"
        if not os.path.isfile(features_path):
            features_h5 = h5py.File(str(features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(
                    test_dataset, desc="Detecting small train features"
                ):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

        ind = 0
        bad_pids = set([])
        features_h5 = h5py.File(features_path, "r")

        ncentroids = 10000
        niter = 20
        verbose = True
        d = self.pid2mean_desc.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(self.pid2mean_desc)
        _, cluster_ind = kmeans.index.search(self.pid2mean_desc, 1)
        cluster_ind2 = cluster_ind.flatten()
        cluster_coord_var = np.zeros(ncentroids)
        for id2 in tqdm(range(ncentroids)):
            mask = cluster_ind2 == id2
            coords = self.xyz_arr[mask]
            cluster_coord_var[id2] = np.mean(np.var(coords, 0))

        import kmeans1d

        mask3 = np.array(kmeans1d.cluster(cluster_coord_var, 2).clusters) == 0
        cluster_list1 = np.arange(ncentroids)[mask3]
        cluster_list2 = np.arange(ncentroids)[np.bitwise_not(mask3)]
        mask41 = np.isin(cluster_ind2, cluster_list1)
        mask42 = np.isin(cluster_ind2, cluster_list2)

        xyz_22 = self.xyz_arr[cluster_ind.flatten() == 8741]
        import open3d as o3d

        point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(self.xyz_arr[mask41])
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()
        good_pids = np.arange(self.xyz_arr.shape[0])[mask41]
        ind = 0
        for example in test_dataset:
            keypoints, descriptors = dd_utils.read_kp_and_desc(example[1], features_h5)

            uv_arr, xyz_pred, pid_list = self.legal_predict(
                keypoints,
                descriptors,
                gpu_index_flat,
                return_pid=True,
            )
            mask0 = np.isin(pid_list, good_pids)
            camera = example[6]

            res = pycolmap.absolute_pose_estimation(
                uv_arr,
                xyz_pred,
                camera,
            )
            t_err0 = float(
                torch.norm(example[4][0:3, 3] - res["cam_from_world"].translation)
            )
            res = pycolmap.absolute_pose_estimation(
                uv_arr[mask0],
                xyz_pred[mask0],
                camera,
            )
            t_err = float(
                torch.norm(example[4][0:3, 3] - res["cam_from_world"].translation)
            )
            print(t_err0, t_err)
            ind += 1
            if ind > 10:
                break
            if t_err < 1:
                mask = np.bitwise_not(res["inliers"])
                for pid in pid_list[mask]:
                    bad_pids.add(pid)

        features_h5.close()

        distances, feature_indices = gpu_index_flat.search(self.pid2mean_desc, 2)

        # import open3d as o3d
        # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz_arr[list(bad_pids)]))
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(width=1920, height=1025)
        # vis.add_geometry(point_cloud)
        # vis.run()
        # vis.destroy_window()
        return

    def evaluate(self):
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

        if self.using_global_descriptors:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}_{self.global_desc_model_name}.txt",
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
                    image_descriptor = np.array(
                        global_features_h5[name]["global_descriptor"]
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
                res = pycolmap.absolute_pose_estimation(
                    uv_arr,
                    xyz_pred,
                    camera,
                    refinement_options={"max_num_iterations": 100},
                )
                # res2 = pycolmap.absolute_pose_estimation(
                #     uv_arr,
                #     xyz_pred,
                #     camera,
                # )
                mat = res["cam_from_world"]
                qvec = " ".join(map(str, mat.rotation.quat[[3, 0, 1, 2]]))
                tvec = " ".join(map(str, mat.translation))
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
                        descriptors = 0.5 * (
                            descriptors + image_descriptor[: descriptors.shape[1]]
                        )

                    uv_arr, xyz_pred = self.legal_predict(
                        keypoints,
                        descriptors,
                        gpu_index_flat,
                    )

                    camera = example[6]
                    # res = pycolmap.absolute_pose_estimation(
                    #     uv_arr,
                    #     xyz_pred,
                    #     camera,
                    # )

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

                    # mat = res["cam_from_world"]
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
        if self.using_global_descriptors:
            file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
            file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}.npy"
        else:
            file_name1 = (
                f"output/{self.ds_name}/codebook_{self.local_desc_model_name}.npy"
            )
            file_name2 = (
                f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}.npy"
            )

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
                    selected_descriptors = 0.5 * (
                        selected_descriptors + image_descriptor[: descriptors.shape[1]]
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
        with torch.no_grad():
            for example in tqdm(testset, desc="Computing pose for test set"):
                name = "/".join(example[1].split("/")[-2:])
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
        return median_tErr, median_rErr


class ConcatenateTrainer(BaseTrainer):
    def collect_image_descriptors(self):
        file_name1 = (
            f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_all.npy"
        )
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}_all.npy"
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
            image2desc[name] = all_desc[idx]
        return image2desc

    def collect_descriptors(self, vis=False):
        print("Using child method")
        file_name1 = f"output/{self.ds_name}/codebook_{self.local_desc_model_name}_{self.global_desc_model_name}_concat.npy"
        file_name2 = f"output/{self.ds_name}/all_pids_{self.local_desc_model_name}_{self.global_desc_model_name}_concat.npy"
        file_name3 = f"output/{self.ds_name}/pid2ind_{self.local_desc_model_name}_{self.global_desc_model_name}_concat.pkl"

        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )
        print(f"Checking if {file_name1} exists")
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

                image_descriptor = self.image2desc[example[1]]
                selected_descriptors = descriptors[idx_arr]
                g1 = np.sqrt(0.5 / selected_descriptors.shape[1])
                g2 = np.sqrt(0.5 / image_descriptor.shape[0])
                selected_descriptors = np.hstack(
                    (
                        selected_descriptors * g1,
                        np.tile(
                            image_descriptor * g2, (selected_descriptors.shape[0], 1)
                        ),
                    )
                )

                for idx, pid in enumerate(selected_pid[ind2]):
                    pid2descriptors.setdefault(pid, []).append(
                        selected_descriptors[idx]
                    )

            features_h5.close()
            all_pid = list(pid2descriptors.keys())
            all_pid = np.array(all_pid)
            pid2mean_desc = np.zeros(
                (
                    len(self.dataset.recon_points),
                    self.feature_dim,
                ),
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

    def legal_predict_with_img_desc(
        self,
        uv_arr,
        features_ori,
        gpu_index_flat,
        img_desc,
    ):
        distances, feature_indices = gpu_index_flat.search(features_ori, 10)
        pid2global_desc = {}
        res2 = []
        for uv_id, ind_arr in enumerate(feature_indices):
            pid_arr = [self.ind2pid[ind] for ind in ind_arr]
            all_desc_np = np.zeros(
                (len(pid_arr), self.global_feature_dim), img_desc.dtype
            )
            for ind1, pid in enumerate(pid_arr):
                if pid not in pid2global_desc:
                    image_ids = [
                        self.dataset.recon_images[img_id].name
                        for img_id in self.dataset.pid2images[pid]
                    ]
                    all_desc = [
                        self.image2desc[f"{self.dataset.images_dir_str}/{img_id}"]
                        for img_id in image_ids
                    ]
                    global_desc = np.mean(all_desc, 0)
                    pid2global_desc[pid] = global_desc
                else:
                    global_desc = pid2global_desc[pid]

                all_desc_np[ind1] = global_desc
            diff2 = np.mean(np.abs(all_desc_np - img_desc), 1)
            pid_wanted = pid_arr[np.argmin(diff2)]
            res2.append(pid_wanted)

        pred_scene_coords_b3 = np.array(
            [self.dataset.recon_points[pid].xyz for pid in res2]
        )
        return uv_arr, pred_scene_coords_b3

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])

        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}_{self.global_desc_model_name}_cc.txt",
            "w",
        )

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

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)
                image_descriptor = dd_utils.read_global_desc(name, global_features_h5)

                g1 = np.sqrt(0.5 / descriptors.shape[1])
                g2 = np.sqrt(0.5 / image_descriptor.shape[0])
                descriptors = np.hstack(
                    (
                        descriptors * g1,
                        np.tile(image_descriptor * g2, (descriptors.shape[0], 1)),
                    )
                )

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )
                camera = example[6]
                res = pycolmap.absolute_pose_estimation(uv_arr, xyz_pred, camera)
                mat = res["cam_from_world"]
                qvec = " ".join(map(str, mat.rotation.quat[[3, 0, 1, 2]]))
                tvec = " ".join(map(str, mat.translation))
                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        features_h5.close()
        result_file.close()
        global_features_h5.close()


class MeanOfLocalDescriptorsTrainer(BaseTrainer):
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
    ):
        super().__init__(
            train_ds,
            test_ds,
            feature_dim,
            global_feature_dim,
            local_desc_model,
            global_desc_model,
            local_desc_conf,
            global_desc_conf,
            False,
            collect_code_book=False,
            run_local_feature_detection_on_test_set=True,
        )
        self.all_global_descriptors = None
        self.all_image_names = None
        self.image2desc = self.collect_image_descriptors()

    def collect_image_descriptors(self):
        file_name1 = (
            f"output/{self.ds_name}/image_desc_{self.local_desc_model_name}.npy"
        )
        file_name2 = (
            f"output/{self.ds_name}/image_desc_name_{self.local_desc_model_name}.npy"
        )
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            features_path = (
                f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
            )
            if not os.path.isfile(features_path):
                features_h5 = h5py.File(str(features_path), "a", libver="latest")
                with torch.no_grad():
                    for example in tqdm(self.dataset, desc="Detecting features"):
                        self.produce_local_descriptors(example[1], features_h5)
                features_h5.close()

            all_desc = np.zeros((len(self.dataset), self.global_feature_dim))
            all_names = []
            idx = 0
            features_h5 = h5py.File(features_path, "r")
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collecting image descriptors"):
                    keypoints, descriptors = dd_utils.read_kp_and_desc(
                        example[1], features_h5
                    )
                    image_descriptor = np.mean(descriptors, 0)
                    all_desc[idx] = image_descriptor
                    all_names.append(example[1])
                    idx += 1
            features_h5.close()
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        image2desc = {}
        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx, : self.feature_dim]
        self.all_global_descriptors = all_desc
        self.all_image_names = all_names
        return image2desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.all_global_descriptors)
        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}_global_only.txt",
            "w",
        )
        print(f"Evaluating with {self.feature_dim}-D descriptors.")
        features_h5 = h5py.File(self.test_features_path, "r")

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)

                image_descriptor = np.mean(descriptors, 0)

                distances, image_ind = gpu_index_flat.search(
                    np.expand_dims(image_descriptor, 0), 1
                )
                db_name = self.all_image_names[image_ind[0][0]]
                name = db_name.split(f"{str(self.dataset.images_dir)}/")[-1]
                db_img_id = self.dataset.image_name2id[name]
                res = self.dataset.recon_images[db_img_id]
                qvec = " ".join(map(str, res.qvec))
                tvec = " ".join(map(str, res.tvec))

                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        result_file.close()
        features_h5.close()


class GlobalDescriptorOnlyTrainer(BaseTrainer):
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
    ):
        self.all_global_descriptors = None
        self.all_image_names = None
        super().__init__(
            train_ds,
            test_ds,
            feature_dim,
            global_feature_dim,
            local_desc_model,
            global_desc_model,
            local_desc_conf,
            global_desc_conf,
            False,
            run_local_feature_detection_on_test_set=False,
            collect_code_book=False,
        )
        self.image2desc = self.collect_image_descriptors()

    def collect_image_descriptors(self):
        file_name1 = (
            f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_all.npy"
        )
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}_all.npy"
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
            image2desc[name] = all_desc[idx]
        self.all_global_descriptors = all_desc
        self.all_image_names = all_names
        return image2desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.global_feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.all_global_descriptors)
        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.global_desc_model_name}_global_only.txt",
            "w",
        )
        print(f"Evaluating with {self.global_feature_dim}-D descriptors.")
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

        global_features_h5 = h5py.File(global_descriptors_path, "r")

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                image_descriptor = dd_utils.read_global_desc(name, global_features_h5)

                distances, image_ind = gpu_index_flat.search(
                    np.expand_dims(image_descriptor, 0), 1
                )
                db_name = self.all_image_names[image_ind[0][0]]
                name = db_name.split(f"{str(self.dataset.images_dir)}/")[-1]
                db_img_id = self.dataset.image_name2id[name]
                res = self.dataset.recon_images[db_img_id]
                qvec = " ".join(map(str, res.qvec))
                tvec = " ".join(map(str, res.tvec))

                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        result_file.close()
        global_features_h5.close()


class MixVPROnlyTrainer:
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
    ):
        self.dataset = train_ds
        self.test_dataset = test_ds
        self.ds_name = self.dataset.ds_type
        self.all_global_descriptors = None
        self.all_image_names = None
        self.global_feature_dim = global_feature_dim
        self.global_desc_model_name = "MixVPR"
        self.encoder_global = VPRModel(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            },
        ).cuda()

        # state_dict = torch.load(
        #     "../MixVPR/resnet50_MixVPR_128_channels(64)_rows(2).ckpt"
        # )
        state_dict = torch.load(
            "../MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt"
        )
        self.encoder_global.load_state_dict(state_dict)
        self.encoder_global.eval()

        self.collect_image_descriptors()

    def collect_image_descriptors(self):
        file_name1 = (
            f"output/{self.ds_name}/image_desc_{self.global_desc_model_name}_all.npy"
        )
        file_name2 = f"output/{self.ds_name}/image_desc_name_{self.global_desc_model_name}_all.npy"
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
                    image = load_image_mix_vpr(example[1])
                    image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                    image_descriptor = image_descriptor.squeeze().cpu().numpy()

                    all_desc[idx] = image_descriptor
                    all_names.append(example[1])
                    idx += 1
            np.save(file_name1, all_desc)
            with open(file_name2, "wb") as handle:
                pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        image2desc = {}
        for idx, name in enumerate(all_names):
            image2desc[name] = all_desc[idx]
        self.all_global_descriptors = all_desc
        self.all_image_names = all_names
        return image2desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """

        index = faiss.IndexFlatL2(self.global_feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.all_global_descriptors)
        result_file = open(
            f"output/{self.ds_name}/Aachen_v1_1_eval_{self.global_desc_model_name}_global_only.txt",
            "w",
        )
        print(f"Evaluating with {self.global_feature_dim}-D descriptors.")
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
                    image = load_image_mix_vpr(example[1])
                    image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                    image_descriptor = image_descriptor.squeeze().cpu().numpy()

                    name = example[1]
                    dict_ = {"global_descriptor": image_descriptor}
                    dd_utils.write_to_h5_file(global_features_h5, name, dict_)
            global_features_h5.close()

        global_features_h5 = h5py.File(global_descriptors_path, "r")

        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                image_descriptor = dd_utils.read_global_desc(name, global_features_h5)

                distances, image_ind = gpu_index_flat.search(
                    np.expand_dims(image_descriptor, 0), 1
                )
                db_name = self.all_image_names[image_ind[0][0]]
                name = db_name.split(f"{str(self.dataset.images_dir)}/")[-1]
                db_img_id = self.dataset.image_name2id[name]
                res = self.dataset.recon_images[db_img_id]
                qvec = " ".join(map(str, res.qvec))
                tvec = " ".join(map(str, res.tvec))

                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        result_file.close()
        global_features_h5.close()
