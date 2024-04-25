import argparse
import torch
import dd_utils
import faiss
import poselib
import numpy as np
import rerun as rr
import open3d as o3d
import cv2
from tqdm import tqdm
from dataset import RobotCarDataset
from trainer import RobotCarTrainer
from ace_util import read_and_preprocess, project_using_pose
from dd_utils import concat_images_different_sizes


def run_function(
    ds_dir,
    local_desc_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_desc_model, retrieval_model, global_desc_dim
    )
    if using_global_descriptors:
        print(f"Using {local_desc_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_desc_model}")
    train_ds_ = RobotCarDataset(ds_dir=ds_dir)
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=True)
    test_ds_with_gt = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=False)

    trainer_ = RobotCarTrainer(
        train_ds_,
        test_ds_,
        local_desc_dim,
        global_desc_dim,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        False,
    )
    index = faiss.IndexFlatL2(trainer_.feature_dim)  # build the index
    res = faiss.StandardGpuResources()
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(trainer_.pid2mean_desc)

    total = 0
    for example in tqdm(test_ds_with_gt):
        image, scale = read_and_preprocess(example[1], conf_ns)
        pred = encoder({"image": torch.from_numpy(image).unsqueeze(0).cuda()})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
        descriptors = pred["descriptors"].T

        nb_matches = 1
        distances, feature_indices = gpu_index_flat.search(descriptors, nb_matches)

        uv_arr_pred = project_using_pose(
            example[4].cuda().unsqueeze(0).float(),
            example[5].cuda().unsqueeze(0).float(),
            trainer_.xyz_arr[feature_indices.ravel()],
        )

        uv_arr_from_kp = (
            torch.from_numpy(keypoints).repeat(1, nb_matches).reshape(-1, 2)
        )
        err = torch.mean(torch.abs(torch.from_numpy(uv_arr_pred) - uv_arr_from_kp), 1)

        min_data = torch.min(err.reshape((-1, nb_matches)), 1)
        uv_kp_0 = uv_arr_from_kp.reshape((-1, nb_matches, 2))[range(keypoints.shape[0]), min_data.indices, :]
        pid_0 = feature_indices[range(keypoints.shape[0]), min_data.indices]
        mask = min_data.values < 5
        uv_arr, xyz_pred = (
            uv_kp_0.numpy(),
            trainer_.xyz_arr[pid_0],
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
        mask = info["inliers"]

        cam = o3d.geometry.LineSet.create_camera_visualization(1024, 1024,
                                                               example[5].double().numpy(),
                                                               example[4].double().inverse().numpy(),
                                                               )
        cam2 = o3d.geometry.LineSet.create_camera_visualization(1024, 1024,
                                                               example[5].double().numpy(),
                                                               np.vstack([pose.Rt, [0, 0, 0, 1]]),
                                                               )
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)
        vis.add_geometry(cam)
        vis.add_geometry(cam2)
        vis.run()
        vis.destroy_window()

        pid2images = {}
        pid_need = set(pid_0[mask])
        for image in tqdm(train_ds_.image2uvs):
            uvs = train_ds_.image2uvs[image]
            pids = train_ds_.image2points[image]
            for idx, pid in enumerate(pids):
                if pid in pid_need:
                    pid2images.setdefault(pid, []).append([image, uvs[idx]])

        for idx in range(uv_arr[mask].shape[0]):
            image0 = cv2.imread(example[1])
            cv2.circle(image0, uv_arr[idx].astype(int), 20, (255, 0, 0), -1)
            images = [image0]
            for img_id, (u, v) in pid2images[pid_0[mask][idx]]:
                image1 = cv2.imread(f"datasets/robotcar/images{train_ds_.image2name[img_id][1:]}".replace("png", "jpg"))
                cv2.circle(image1, (int(u), int(v)), 20, (255, 0, 0), -1)
                images.append(image1)
                if len(images) > 10:
                    break
            image = concat_images_different_sizes(images)
            cv2.imwrite(f"debug/test{idx}.png", image)
        break

        _, feature_indices = gpu_index_flat.search(descriptors, 1)
        uv_arr, xyz_pred = (
            keypoints,
            trainer_.xyz_arr[feature_indices.ravel()],
        )

        pose2, _ = poselib.estimate_absolute_pose(
            uv_arr,
            xyz_pred,
            camera_dict,
        )

        pose_err = torch.mean(torch.abs(example[4][:3, 3] - pose.t))
        total += pose_err
        tqdm.write(f"{example[1]}, {pose_err}")
    print(total / len(test_ds_with_gt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument("--use_global", type=int, default=1)

    parser.add_argument(
        "--local_desc",
        type=str,
        default="r2d2",
    )
    parser.add_argument(
        "--local_desc_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="eigenplaces",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    run_function(
        args.dataset,
        args.local_desc,
        args.global_desc,
        int(args.local_desc_dim),
        int(args.global_desc_dim),
        bool(args.use_global),
    )
