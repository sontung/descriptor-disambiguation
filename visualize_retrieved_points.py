import argparse
from pathlib import Path
import numpy as np
import rerun as rr
import dd_utils
from clustering import reduce_map_using_min_cover
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer
import open3d as o3d
import cv2
import torch
from trainer import project_using_pose
from tqdm import tqdm


def compute_reproj_err(gt_pose, xyz, uv):
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = 1670.480625
    intrinsics[1, 1] = 1670.480625
    intrinsics[0, 2] = 960.0
    intrinsics[1, 2] = 540.0
    uv_arr_pred = project_using_pose(
        torch.from_numpy(gt_pose).cuda().unsqueeze(0).float(),
        intrinsics.cuda().unsqueeze(0).float(),
        xyz,
    )
    m1 = uv_arr_pred[:, 0] > 0
    m2 = uv_arr_pred[:, 1] > 0
    m3 = uv_arr_pred[:, 0] < 1920
    m4 = uv_arr_pred[:, 1] < 1080
    oob = np.all([m1, m2, m3, m4], 0)
    return np.mean(np.abs(uv-uv_arr_pred), 1), oob, uv_arr_pred


def make_pic(good_result, bad_result, res_name, rgb_arr):
    (
        name1,
        t_err1,
        r_err1,
        uv_arr1,
        xyz_pred1,
        pose1,
        gt_pose1,
        mask1,
        pid_list1,
    ) = good_result
    (
        name2,
        t_err2,
        r_err2,
        uv_arr2,
        xyz_pred2,
        pose2,
        gt_pose2,
        mask2,
        pid_list2,
    ) = bad_result

    gt_pose1 = dd_utils.return_pose_mat_no_inv(gt_pose1.qvec, gt_pose1.tvec)
    gt_pose2 = dd_utils.return_pose_mat_no_inv(gt_pose2.qvec, gt_pose2.tvec)

    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240

    cam1 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, np.vstack([pose1.Rt, [0, 0, 0, 1]]), scale=9
    )
    cam2 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, np.vstack([pose2.Rt, [0, 0, 0, 1]]), scale=9
    )
    cam3 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, gt_pose2, scale=9
    )

    cam1.paint_uniform_color((0, 0, 0))
    cam2.paint_uniform_color((0, 0, 0))
    cam3.paint_uniform_color((0, 1, 0))

    xyz1 = xyz_pred1
    xyz2 = xyz_pred2
    pred1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1))
    pred2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz2))

    dis1, oob1, uv1_pred = compute_reproj_err(gt_pose1, xyz1, uv_arr1)
    dis2, oob2, uv2_pred = compute_reproj_err(gt_pose1, xyz2, uv_arr2)
    colors1 = np.tile([1, 0, 0], (xyz1.shape[0], 1))
    colors1[oob1] = [0, 1, 0]
    colors2 = np.tile([1, 0, 0], (xyz1.shape[0], 1))
    colors2[oob2] = [0, 1, 0]

    pred1.colors = o3d.utility.Vector3dVector(colors1)
    pred2.colors = o3d.utility.Vector3dVector(colors2)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pred2)
    # vis.add_geometry(cam1)
    # vis.add_geometry(cam2)
    # vis.run()
    # vis.destroy_window()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1848, height=1016)
    parameters = o3d.io.read_pinhole_camera_parameters("viewpoints_pc_retrieval.json")
    vis.add_geometry(cam1, reset_bounding_box=True)
    vis.add_geometry(cam3, reset_bounding_box=True)
    vis.add_geometry(cam2, reset_bounding_box=True)
    vis.add_geometry(pred1, reset_bounding_box=True)
    vis.add_geometry(pred2, reset_bounding_box=True)
    vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    vis.remove_geometry(cam2, reset_bounding_box=False)
    vis.remove_geometry(pred2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/good.png", do_render=True)
    vis.remove_geometry(cam1, reset_bounding_box=False)
    vis.remove_geometry(pred1, reset_bounding_box=False)

    vis.add_geometry(cam2, reset_bounding_box=False)
    vis.add_geometry(pred2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/bad.png", do_render=True)
    vis.destroy_window()

    # if t_err1 - t_err2 > 0:
    #
    #     im1 = cv2.imread(f"debug/good.png")
    #     im2 = cv2.imread(f"debug/bad.png")
    #     im3 = cv2.hconcat([im2[150:850, 500:1500], im1[150:850, 500:1500]])
    #     t_err1, t_err2 = map(lambda du: round(du, 2), [t_err1, t_err2])
    #     cv2.imwrite(f"debug/both-{res_name}-{t_err1}-{t_err2}.png", im3)

    return


def visualize_matches(good_results, bad_results, rgb_arr):
    for idx in tqdm(range(len(good_results))):
        idx_str = "{:03d}".format(idx)
        make_pic(good_results[idx], bad_results[idx], idx_str, rgb_arr)
        im1 = cv2.imread(f"debug/good.png")
        im2 = cv2.imread(f"debug/bad.png")
        im3 = cv2.hconcat([im2[150:850, 500:1500], im1[150:850, 500:1500]])
        cv2.imwrite(f"debug/both-{idx_str}.png", im3)
    return


def run_function(
    root_dir_,
    local_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_model, retrieval_model, global_desc_dim
    )
    if using_global_descriptors:
        print(f"Using {local_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_model}")

    # ds_name = "Cambridge_KingsCollege"
    ds_name = "GreatCourt"
    print(f"Processing {ds_name}")
    train_ds_ = CambridgeLandmarksDataset(
        train=True, ds_name=ds_name, root_dir=root_dir_
    )
    test_ds_ = CambridgeLandmarksDataset(
        train=False, ds_name=ds_name, root_dir=f"{root_dir_}"
    )
    # visualize(train_ds_)

    train_ds_2 = CambridgeLandmarksDataset(
        train=True, ds_name=ds_name, root_dir=root_dir_
    )
    # chosen_list = reduce_map_using_min_cover(train_ds_, trainer_.image2pid_via_new_features)

    trainer_ = CambridgeLandmarksTrainer(
        train_ds_2,
        test_ds_,
        local_desc_dim,
        global_desc_dim,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        True,
    )
    trainer_2 = CambridgeLandmarksTrainer(
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

    res = trainer_.process()
    res_bad = trainer_2.process()
    visualize_matches(res, res_bad, trainer_.rgb_arr)

    bad_name_list = [
        "rgb/seq4_frame00093.png",
        "rgb/seq4_frame00091.png",
        "rgb/seq4_frame00086.png",
        "rgb/seq1_frame00421.png",
        "rgb/seq1_frame00440.png",
    ]

    trans, rot, name2err = trainer_.evaluate(return_name2err=True)
    trans2, rot2, name2err2 = trainer_2.evaluate(return_name2err=True)
    all_diff = {}
    all_name = []
    for name in name2err:
        t1, r1 = name2err[name]
        t2, r2 = name2err2[name]
        diff = (t2 - t1) + (r2 - r1)
        all_diff[name] = diff
        all_name.append(name)
    n1 = min(all_name, key=lambda du1: all_diff[du1])
    n2 = max(all_name, key=lambda du1: all_diff[du1])
    print(n1, all_diff[n1], name2err[n1], name2err2[n1])
    print(n2, all_diff[n2], name2err[n2], name2err2[n2])

    all_name_sorted = sorted(all_name, key=lambda du1: all_diff[du1])
    for name in all_name_sorted[:5]:
        print(all_diff[name])
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/cambridge",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument("--use_global", type=int, default=1)
    parser.add_argument(
        "--local_desc",
        type=str,
        default="d2net",
    )
    parser.add_argument(
        "--local_desc_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="mixvpr",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=512,
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
