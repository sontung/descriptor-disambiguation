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
sys.path.append("../CosPlace")
from cosplace_utils import load_image as load_image_cosplace
from cosplace_utils import load_model as load_model_cosplace

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
        self.dataset = AachenDataset()

        self.name2uv = {}
        self.ds_name = "aachen"
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

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

        self.encoder_global = load_model_cosplace(
            "../CosPlace/models/resnet50_128.pth", "ResNet50"
        )

        # model_dict = conf["netvlad"]["model"]
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # Model = dynamic_load(extractors, model_dict["name"])
        # self.encoder_global = Model(model_dict).eval().to(device)
        # conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        # conf_ns_retrieval.resize_max = conf["netvlad"]["preprocessing"]["resize_max"]
        # self.conf_retrieval = conf_ns_retrieval

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
        ) = self.collect_descriptors(conf_ns)
        self.all_ind_in_train_set = np.array(
            [self.pid2ind[pid] for pid in self.all_pid_in_train_set]
        )
        self.ind2pid = {v: k for k, v in self.pid2ind.items()}

    def collect_image_descriptors(self):
        file_name1 = f"output/{self.ds_name}/image_desc.npy"
        file_name2 = f"output/{self.ds_name}/image_desc_name.npy"
        if os.path.isfile(file_name1):
            all_desc = np.load(file_name1)
            afile = open(file_name2, "rb")
            all_names = pickle.load(afile)
            afile.close()
        else:
            all_desc = np.zeros((len(self.dataset), 128))
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

    def produce_image_descriptor(self, name):
        image = load_image_cosplace(name, resize_test_imgs=True)
        image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
        image_descriptor = image_descriptor.squeeze().cpu().numpy()
        return image_descriptor

    def collect_descriptors(self, conf, vis=False):
        file_name1 = f"output/{self.ds_name}/codebook_r2d2.npy"
        file_name2 = f"output/{self.ds_name}/all_pids_r2d2.npy"
        file_name3 = f"output/{self.ds_name}/pid2ind.pkl"
        if os.path.isfile(file_name1):
            pid2mean_desc = np.load(file_name1)
            all_pid = np.load(file_name2)
            afile = open(file_name3, "rb")
            pid2ind = pickle.load(afile)
            afile.close()
        else:
            pid2descriptors = {}
            r2d2_kp_path = Path(f"output/{self.ds_name}/r2d2_kp")
            r2d2_desc_path = Path(f"output/{self.ds_name}/r2d2_desc")
            r2d2_kp_path.mkdir(parents=True, exist_ok=True)
            r2d2_desc_path.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                for example in tqdm(self.dataset, desc="Collect point descriptors"):
                    image, scale = read_and_preprocess(example[1], conf)

                    image_id = example[1].split("/")[-1]
                    kp_file = r2d2_kp_path/f"{image_id}.npy"
                    desc_file = r2d2_desc_path/f"{image_id}.npy"
                    if os.path.isfile(kp_file) and os.path.isfile(desc_file):
                        keypoints = np.load(str(kp_file))
                        descriptors = np.load(str(desc_file))
                    else:
                        pred = self.encoder(
                            {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                        )

                        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                        image_descriptor = self.image2desc[example[1]]

                        keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
                        descriptors = pred["descriptors"].T
                        np.save(str(kp_file), keypoints)
                        np.save(str(desc_file), descriptors)

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

                    # image, _ = read_and_preprocess(example[1], self.conf_retrieval)
                    # image_descriptor = self.encoder_global(
                    #     {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                    # )["global_descriptor"].cpu().numpy()

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
                (len(self.dataset.recon_points), desc_dim),
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

    def evaluate(self):
        """
        write to pose file as name.jpg qw qx qy qz tx ty tz
        :return:
        """
        test_set = AachenDataset(train=False)
        index = faiss.IndexFlatL2(128)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])
        result_file = open(f"output/{self.ds_name}/Aachen_v1_1_eval_dd.txt", "w")
        with torch.no_grad():
            for example in tqdm(test_set, desc="Computing pose for test set"):
                image_name = example[1]
                image, scale = read_and_preprocess(image_name, self.conf)
                pred = self.encoder(
                    {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                )
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
                descriptors = pred["descriptors"].T

                intrinsics_33 = example[5].cpu()
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                # image, _ = read_and_preprocess(example[1], self.conf_retrieval)
                # image_descriptor = self.encoder_global(
                #     {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                # )["global_descriptor"].squeeze().cpu().numpy()

                image_descriptor = self.produce_image_descriptor(example[1])

                # image = load_image_mix_vpr(image_name)
                # image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                # image_descriptor = image_descriptor.squeeze().cpu().numpy()
                descriptors = 0.5 * (descriptors + image_descriptor)

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                # pairs = []
                # for j, (x, y) in enumerate(uv_arr):
                #     xy = [x, y]
                #     xyz = xyz_pred[j]
                #     pairs.append((xy, xyz))

                camera = example[6]
                res = pycolmap.absolute_pose_estimation(uv_arr, xyz_pred, camera)
                qw, qx, qy, qz = res["qvec"]
                tx, ty, tz = res["tvec"]

                # pose, info = localize_pose_lib(pairs, focal_length, ppX, ppY)
                # pose_mat = torch.from_numpy(np.vstack([pose.Rt, np.array([0, 0, 0, 1])])).inverse()
                #
                # qx, qy, qz, qw = Rotation.from_matrix(pose_mat.numpy()[:3, :3]).as_quat()
                # qw, qx, qy, qz = pose.q
                # tx, ty, tz = pose.t
                image_id = example[2].split("/")[-1]
                print(
                    f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz}", file=result_file
                )

    def evaluate_with_hloc(self):
        hloc_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/aachen_v1.1/Aachen-v1.1_hloc_superpoint+superglue_netvlad50.txt"
        with open(hloc_file) as file:
            lines = [line.rstrip() for line in file]
        img2gt = {}
        for line in lines:
            img_id, qw, qx, qy, qz, tx, ty, tz = line.split(" ")
            qw, qx, qy, qz, tx, ty, tz = map(float, [qw, qx, qy, qz, tx, ty, tz])
            img2gt[img_id] = [qw, qx, qy, qz, tx, ty, tz]

        test_set = AachenDataset(train=False)
        index = faiss.IndexFlatL2(128)  # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])
        with torch.no_grad():
            for example in tqdm(test_set, desc="Computing pose for test set"):
                image_name = example[1]
                image, scale = read_and_preprocess(image_name, self.conf)
                pred = self.encoder(
                    {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                )
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                keypoints = pred["keypoints"]
                descriptors = pred["descriptors"].T
                keypoints /= scale

                intrinsics_33 = example[5].cpu()
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                image = load_image_mix_vpr(image_name)
                image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                image_descriptor = image_descriptor.squeeze().cpu().numpy()

                descriptors = 0.5 * (descriptors + image_descriptor)

                uv_arr, xyz_pred = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                pairs = []
                for j, (x, y) in enumerate(uv_arr):
                    xy = [x, y]
                    xyz = xyz_pred[j]
                    pairs.append((xy, xyz))

                camera = example[6]
                res = pycolmap.absolute_pose_estimation(uv_arr, xyz_pred, camera)
                # qx, qy, qz, qw = res["qvec"]
                # tx, ty, tz = res["tvec"]

                pose, info = localize_pose_lib(pairs, focal_length, ppX, ppY)

                image_id = example[2].split("/")[-1]
                qw, qx, qy, qz, tx, ty, tz = img2gt[image_id]
                pose_q = np.array([qx, qy, qz, qw])
                pose_R = Rotation.from_quat(pose_q).as_matrix()

                gt_pose = np.identity(4)
                gt_pose[0:3, 0:3] = pose_R
                gt_pose[0:3, 3] = [tx, ty, tz]
                gt_pose = torch.from_numpy(gt_pose)

                est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
                est_pose = np.linalg.inv(est_pose)
                out_pose = torch.from_numpy(est_pose)

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

                gt_R = gt_pose[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi
                print()

    def test_model(self):
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
        gpu_index_flat.add(self.pid2mean_desc[self.all_ind_in_train_set])

        with torch.no_grad():
            for example in tqdm(self.dataset, desc="Computing pose for test set"):
                image_name = example[1]
                image, scale = read_and_preprocess(image_name, self.conf)
                pred = self.encoder(
                    {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                )
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
                descriptors = pred["descriptors"].T

                intrinsics_33 = example[5].cpu()
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                image = load_image_mix_vpr(image_name)
                image_descriptor = self.encoder_global(image.unsqueeze(0).cuda())
                image_descriptor = image_descriptor.squeeze().cpu().numpy()

                descriptors = 0.5 * (descriptors + image_descriptor)

                uv_arr2, xyz_pred2 = self.legal_predict(
                    keypoints,
                    descriptors,
                    gpu_index_flat,
                )

                gt_pose_B44 = example[4].inverse().unsqueeze(0)
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
        assert total_frames == len(self.dataset)

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
    trainer = TrainerACE()
    # trainer.evaluate_with_hloc()
    trainer.evaluate()
    # trainer.test_model()
    # trainer.train()
