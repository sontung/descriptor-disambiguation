import math
import os
import pickle
import sys
from pathlib import Path
import kmeans1d
import cv2
import faiss
import h5py
import hurry.filesize
import numpy as np
import poselib
import torch
from pykdtree.kdtree import KDTree
from tqdm import tqdm
import kornia
import dd_utils
from ace_util import read_and_preprocess, project_using_pose


def retrieve_pid(pid_list, uv_gt, keypoints):
    tree = KDTree(keypoints.astype(uv_gt.dtype))
    dis, ind = tree.query(uv_gt)
    mask = dis < 5
    selected_pid = np.array(pid_list)[mask]
    return selected_pid, mask, ind


def compute_pose_error(pose, pose_gt):
    R_gt, t_gt = pose_gt.qvec2rotmat(), pose_gt.tvec
    R, t = pose.R, pose.t
    t_err = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
    r_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, r_err


def combine_descriptors(local_desc, global_desc, lambda_value_, until=None):
    if until is None:
        until = local_desc.shape[1]
    res = lambda_value_ * local_desc + (1 - lambda_value_) * global_desc[:until]
    return res


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
        collect_code_book=True,
        lambda_val=0.5,
        convert_to_db_desc=True,
        codebook_dtype=np.float16,
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

        try:
            self.local_desc_model_name = local_desc_model.conf["name"]
        except AttributeError:
            if type(local_desc_model) == tuple:
                self.local_desc_model_name = "sdf2"
            elif type(local_desc_model) == kornia.feature.dedode.dedode.DeDoDe:
                self.local_desc_model_name = "dedode"

        self.global_desc_model_name = (
            f"{global_desc_model.conf['name']}_{global_feature_dim}"
        )
        self.local_desc_model = local_desc_model
        self.local_desc_conf = local_desc_conf

        self.global_desc_model = global_desc_model
        self.global_desc_conf = global_desc_conf

        self.test_features_path = None
        self.rgb_arr = None
        self.pca = None
        self.using_pca = False
        self.lambda_val = lambda_val
        print(f"using lambda val={self.lambda_val}")
        self.global_desc_mean = 0
        self.global_desc_std = 1

        self.local_features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )
        self.convert_to_db_desc = convert_to_db_desc
        self.all_names = []
        self.all_image_desc = None
        if self.using_global_descriptors:
            self.image2desc = self.collect_image_descriptors()
        else:
            self.image2desc = {}

        self.xyz_arr = None
        self.map_reduction = False
        self.codebook_dtype = codebook_dtype
        self.total_diff = np.zeros(self.global_feature_dim)
        self.count = 0
        self.special_pid_list = None
        self.image2pid_via_new_features = {}
        self.pid2mean_desc_vanilla = None
        self.centroids_for_image_desc = None
        self.all_outlier_descs = None
        if collect_code_book:
            self.pid2descriptors = {}
            self.pid2count = {}
            self.pid2mean_desc = self.collect_descriptors()
        else:
            self.pid2mean_desc = None
            self.all_pid_in_train_set = None
            self.pid2ind = None
            self.all_ind_in_train_set = None
            self.ind2pid = None

    def load_local_features(self):
        features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_train.h5"
        )

        if not os.path.isfile(features_path):
            features_h5 = h5py.File(str(features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Detecting features"):
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()
        features_h5 = h5py.File(features_path, "r")
        return features_h5

    def detect_local_features_on_test_set(self):
        self.test_features_path = (
            f"output/{self.ds_name}/{self.local_desc_model_name}_features_test.h5"
        )
        if not os.path.isfile(self.test_features_path):
            features_h5 = h5py.File(str(self.test_features_path), "a", libver="latest")
            with torch.no_grad():
                for example in tqdm(
                    self.test_dataset, desc="Detecting testing features"
                ):
                    if example is None:
                        continue
                    self.produce_local_descriptors(example[1], features_h5)
            features_h5.close()

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
            print(f"Cannot find {file_name1}")
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
        # if self.convert_to_db_desc:
        self.all_image_desc = all_desc
        self.all_names = all_names
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

    def produce_local_descriptors(self, name, fd=None):
        image, scale = read_and_preprocess(name, self.local_desc_conf)
        if self.local_desc_model_name == "sfd2":
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
        elif self.local_desc_model_name == "dedode":
            keypoints, scores, descriptors = self.local_desc_model(
                torch.from_numpy(image).float().unsqueeze(0).cuda()
            )
            assert scale == 1
            pred = {
                "keypoints": keypoints.squeeze().cpu().numpy(),
                "descriptors": descriptors.squeeze().cpu().numpy().T,
            }
            # dense = self.local_desc_model.describe(
            #     torch.from_numpy(image).float().unsqueeze(0).cuda()
            # )
            # image2 = image*255
            # image2 = image2.astype(np.uint8).transpose((1, 2, 0))
            # for u, v in keypoints[0].int().cpu().numpy():
            #     cv2.circle(image2, (u, v), 5, (0, 255, 0))
            # cv2.imwrite("debug/img.png", image2)
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
        if fd is None:
            return dict_
        dd_utils.write_to_h5_file(fd, name, dict_)

    def collect_descriptors_loop(
        self, features_h5, pid2mean_desc, pid2count, using_global_desc, id_list=None
    ):
        pid2ind = {}
        index_for_array = -1
        self.image2pid_via_new_features = {}
        for example_id, example in enumerate(
            tqdm(self.dataset, desc="Collecting point descriptors")
        ):
            keypoints, descriptors = dd_utils.read_kp_and_desc(example[1], features_h5)

            pid_list = example[3]
            uv = example[-1]

            selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
            selected_descriptors = descriptors[ind[mask]]
            if using_global_desc:
                image_descriptor = self.image2desc[example[1]]
                selected_descriptors = combine_descriptors(
                    selected_descriptors, image_descriptor, self.lambda_val
                )

            for idx, pid in enumerate(selected_pid):
                if pid not in pid2ind:
                    index_for_array += 1
                    pid2ind[pid] = index_for_array

            idx2 = [pid2ind[pid] for pid in selected_pid]
            pid2mean_desc[idx2] += selected_descriptors
            pid2count[idx2] += 1
        index_for_array += 1
        pid2mean_desc = pid2mean_desc[:index_for_array, :] / pid2count[
            :index_for_array
        ].reshape(-1, 1)
        if pid2mean_desc.dtype != self.codebook_dtype:
            pid2mean_desc = pid2mean_desc.astype(self.codebook_dtype)

        if np.sum(np.isnan(pid2mean_desc)) > 0:
            print(f"NaN detected in codebook: {np.sum(np.isnan(pid2mean_desc))}")

        print(f"Codebook size: {round(sys.getsizeof(pid2mean_desc) / 1e9, 2)} GB")
        print(f"Codebook dtype: {pid2mean_desc.dtype}")
        return pid2mean_desc, pid2ind

    def collect_descriptors(self, vis=False):
        features_h5 = self.load_local_features()

        pid2mean_desc = np.zeros(
            (len(self.dataset.recon_points), self.feature_dim),
            self.codebook_dtype,
        )
        pid2count = np.zeros(len(self.dataset.recon_points))

        pid2mean_desc, pid2ind = self.collect_descriptors_loop(
            features_h5, pid2mean_desc, pid2count, self.using_global_descriptors
        )
        print(pid2mean_desc.shape)

        self.xyz_arr = np.zeros((pid2mean_desc.shape[0], 3))
        self.pid2ind = pid2ind
        for pid in pid2ind:
            self.xyz_arr[pid2ind[pid]] = self.dataset.recon_points[pid].xyz

        np.save(
            f"output/{self.ds_name}/codebook-{self.local_desc_model_name}-{self.global_desc_model_name}.npy",
            pid2mean_desc,
        )
        features_h5.close()

        return pid2mean_desc

    def process_descriptor(
        self, name, features_h5, global_features_h5, gpu_index_flat_for_image_desc=None
    ):
        keypoints, descriptors = dd_utils.read_kp_and_desc(name, features_h5)

        if self.using_global_descriptors:
            image_descriptor = dd_utils.read_global_desc(name, global_features_h5)

            if self.convert_to_db_desc:
                _, ind = gpu_index_flat_for_image_desc.search(
                    image_descriptor.reshape(1, -1), 1
                )
                image_descriptor = self.all_image_desc[int(ind)]

            descriptors = combine_descriptors(
                descriptors, image_descriptor, self.lambda_val
            )
        return keypoints, descriptors

    def return_faiss_indices(self):
        index = faiss.IndexFlatL2(self.feature_dim)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc)
        if self.convert_to_db_desc and self.using_global_descriptors:
            index2 = faiss.IndexFlatL2(self.global_feature_dim)  # build the index
            res2 = faiss.StandardGpuResources()
            gpu_index_flat_for_image_desc = faiss.index_cpu_to_gpu(res2, 0, index2)
            gpu_index_flat_for_image_desc.add(self.all_image_desc)
            print("Converting to DB descriptors")
            print(
                f"DB desc size: {hurry.filesize.size(sys.getsizeof(self.all_image_desc))}"
            )
        else:
            gpu_index_flat_for_image_desc = None
        return gpu_index_flat, gpu_index_flat_for_image_desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """
        self.ind2pid = {ind: pid for pid, ind in self.pid2ind.items()}
        self.detect_local_features_on_test_set()
        gpu_index_flat, gpu_index_flat_for_image_desc = self.return_faiss_indices()

        if self.using_global_descriptors:
            result_file = open(
                f"output/{self.ds_name}/Aachen_v1_1_eval_{self.local_desc_model_name}_{self.global_desc_model_name}_{self.global_feature_dim}_{self.lambda_val}_{self.convert_to_db_desc}.txt",
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
                keypoints, descriptors = self.process_descriptor(
                    name, features_h5, global_features_h5, gpu_index_flat_for_image_desc
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

                image_id = example[2].split("/")[-1]
                print(f"{image_id} {qvec} {tvec}", file=result_file)
        features_h5.close()
        result_file.close()
        global_features_h5.close()

    def legal_predict(
        self,
        uv_arr,
        features_ori,
        gpu_index_flat,
        remove_duplicate=False,
        return_indices=True,
        ratio_test=False,
    ):
        if ratio_test:
            distances, feature_indices = gpu_index_flat.search(
                features_ori.astype(self.codebook_dtype), 2
            )
            mask = distances[:, 0] / distances[:, 1] < 0.8
            feature_indices = feature_indices[mask][:, 0]
            uv_arr = uv_arr[mask]
        else:
            distances, feature_indices = gpu_index_flat.search(
                features_ori.astype(self.codebook_dtype), 1
            )

        feature_indices = feature_indices.ravel()
        if self.special_pid_list is not None:
            mask = [
                True if self.ind2pid[ind] in self.special_pid_list else False
                for ind in feature_indices
            ]
            feature_indices = feature_indices[mask]
            uv_arr = uv_arr[mask]

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

        if return_indices:
            return uv_arr, pred_scene_coords_b3, feature_indices

        return uv_arr, pred_scene_coords_b3


class RobotCarTrainer(BaseTrainer):
    def reduce_map_size(self):
        if self.map_reduction:
            return
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
            self.dataset.image2uvs[img] = np.array(self.dataset.image2uvs[img])[mask]
        self.dataset.image2points = img2points2
        self.map_reduction = True
        return inlier_ind

    def collect_descriptors(self, vis=False):
        file_name1 = f"output/{self.ds_name}/pid2mean_desc_debug.npy"
        file_name2 = f"output/{self.ds_name}/pid2ind_debug.pkl"
        if os.path.isfile(file_name1) and os.path.isfile(file_name2):
            pid2mean_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            pid2ind = pickle.load(afile)
            afile.close()
        else:
            print(f"Cant find {file_name1} {file_name2}")
            features_h5 = self.load_local_features()
            pid2mean_desc = np.zeros(
                (self.dataset.xyz_arr.shape[0], self.feature_dim),
                np.float64,
            )
            pid2count = np.zeros(self.dataset.xyz_arr.shape[0], self.codebook_dtype)

            pid2mean_desc, pid2ind = self.collect_descriptors_loop(
                features_h5, pid2mean_desc, pid2count, self.using_global_descriptors
            )
            features_h5.close()

            np.save(
                file_name1,
                pid2mean_desc,
            )
            with open(file_name2, "wb") as handle:
                pickle.dump(pid2ind, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.xyz_arr = np.zeros((pid2mean_desc.shape[0], 3))
        for pid in pid2ind:
            self.xyz_arr[pid2ind[pid]] = self.dataset.xyz_arr[pid]
        self.pid2ind = pid2ind

        return pid2mean_desc

    def evaluate(self):
        self.detect_local_features_on_test_set()
        gpu_index_flat, gpu_index_flat_for_image_desc = self.return_faiss_indices()

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
                f"output/{self.ds_name}/RobotCar_eval_"
                f"{self.local_desc_model_name}_"
                f"{self.global_desc_model_name}_"
                f"{self.global_feature_dim}_"
                f"{self.lambda_val}_"
                f"{self.convert_to_db_desc}.txt",
                "w",
            )
        else:
            result_file = open(
                f"output/{self.ds_name}/RobotCar_eval_{self.local_desc_model_name}.txt",
                "w",
            )

        pgt_matches = h5py.File(f"outputs/{self.ds_name}/matches2d_3d.h5", "r")
        ind2pid = {ind: pid for pid, ind in self.pid2ind.items()}
        mean_acc = []
        with torch.no_grad():
            for example in tqdm(self.test_dataset, desc="Computing pose for test set"):
                name = example[1]
                image_name_wo_dir = name.split(self.dataset.images_dir_str)[-1][1:]
                keypoints, descriptors = self.process_descriptor(
                    name, features_h5, global_features_h5, gpu_index_flat_for_image_desc
                )

                uv_arr, xyz_pred, indices = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                data = pgt_matches[image_name_wo_dir]
                uv_arr_pgt = np.array(data["uv"])
                pid_list_pgt = np.array(data["pid"])
                tree = KDTree(uv_arr_pgt)
                dis, ind_sub1 = tree.query(uv_arr, 1)
                mask = dis < 1
                pid_list_pred = np.array([ind2pid[ind] for ind in indices[mask]])
                diff = pid_list_pred-pid_list_pgt[ind_sub1[mask]]
                acc = np.sum(diff==0)/diff.shape[0]
                mean_acc.append(acc)

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
        print(np.mean(mean_acc))


class CMUTrainer(BaseTrainer):
    def clear(self):
        del self.pid2mean_desc

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """
        self.detect_local_features_on_test_set()
        gpu_index_flat, gpu_index_flat_for_image_desc = self.return_faiss_indices()

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
            result_file_name = (
                f"output/{self.ds_name}/CMU_eval"
                f"_{self.local_desc_model_name}_"
                f"{self.global_desc_model_name}_"
                f"{self.lambda_val}_"
                f"{self.convert_to_db_desc}.txt"
            )
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
                    keypoints, descriptors = self.process_descriptor(
                        name,
                        features_h5,
                        global_features_h5,
                        gpu_index_flat_for_image_desc,
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


class CambridgeLandmarksTrainer(BaseTrainer):
    def evaluate(self, return_name2err=False):
        self.ind2pid = {ind: pid for pid, ind in self.pid2ind.items()}
        self.detect_local_features_on_test_set()
        gpu_index_flat, gpu_index_flat_for_image_desc = self.return_faiss_indices()

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
        testset = self.test_dataset
        name2err = {}
        rErrs = []
        tErrs = []
        with torch.no_grad():
            for example in tqdm(testset, desc="Computing pose for test set"):
                name = "/".join(example[1].split("/")[-2:])

                keypoints, descriptors = self.process_descriptor(
                    name, features_h5, global_features_h5, gpu_index_flat_for_image_desc
                )

                uv_arr, xyz_pred, xyz_indices, desc_distances = self.legal_predict(
                    keypoints, descriptors, gpu_index_flat, return_indices=True
                )

                camera = example[6]
                camera_dict = {
                    "model": camera.model.name,
                    "height": camera.height,
                    "width": camera.width,
                    "params": camera.params,
                }
                pose0, info = poselib.estimate_absolute_pose(
                    uv_arr,
                    xyz_pred,
                    camera_dict,
                )
                t_err0, r_err = compute_pose_error(pose0, example[4])
                tErrs.append(t_err0 * 100)
                rErrs.append(r_err)

        features_h5.close()
        global_features_h5.close()

        # Compute median errors.
        median_rErr = np.median(rErrs)
        median_tErr = np.median(tErrs)
        if return_name2err:
            return median_tErr, median_rErr, name2err
        return median_tErr, median_rErr

    def process(self):
        self.detect_local_features_on_test_set()
        gpu_index_flat, gpu_index_flat_for_image_desc = self.return_faiss_indices()

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
        testset = self.test_dataset
        res = []
        with torch.no_grad():
            for example in tqdm(testset, desc="Computing pose for test set"):
                name = "/".join(example[1].split("/")[-2:])
                keypoints, descriptors = self.process_descriptor(
                    name, features_h5, global_features_h5, gpu_index_flat_for_image_desc
                )

                uv_arr, xyz_pred, pid_list = self.legal_predict(
                    keypoints, descriptors, gpu_index_flat, return_indices=True
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

                t_err, r_err = compute_pose_error(pose, example[4])

                res.append(
                    [
                        name,
                        t_err,
                        r_err,
                        uv_arr,
                        xyz_pred,
                        pose,
                        example[4],
                        info["inliers"],
                        pid_list,
                    ]
                )
                # break

        features_h5.close()
        global_features_h5.close()

        return res