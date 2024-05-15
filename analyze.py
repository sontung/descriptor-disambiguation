import argparse
from pathlib import Path
import numpy as np
import rerun as rr
import dd_utils
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer
import open3d as o3d
import cv2


def visualize(ds):
    rr.init("rerun_example_app")

    rr.connect()  # Connect to a remote viewer
    rr.spawn()  # Spawn a child process with a viewer and connect
    # rr.save("recording.rrd")  # Stream all logs to disk

    # Associate subsequent data with 42 on the “frame” timeline
    rr.set_time_sequence("frame", 42)

    # Log colored 3D points to the entity at `path/to/points`

    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds.xyz_arr))
    cl, inlier_ind = point_cloud.remove_radius_outlier(nb_points=16, radius=5)
    rr.log(
        "path/to/points",
        rr.Points3D(ds.xyz_arr[inlier_ind], colors=ds.rgb_arr[inlier_ind] / 255),
    )


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
    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240

    cam1 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, np.vstack([pose1.Rt, [0, 0, 0, 1]]), scale=7
    )
    cam2 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, np.vstack([pose2.Rt, [0, 0, 0, 1]]), scale=7
    )
    cam3 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, gt_pose2.numpy(), scale=7
    )

    cam1.paint_uniform_color((0, 0, 0))
    cam2.paint_uniform_color((0, 0, 0))
    cam3.paint_uniform_color((0, 1, 0))

    # ori1 = o3d.geometry.PointCloud(
    #     o3d.utility.Vector3dVector(
    #         np.repeat(cam1.get_center().reshape(1, -1), 3, axis=0).reshape(-1, 3)
    #     )
    # )
    # ori2 = o3d.geometry.PointCloud(
    #     o3d.utility.Vector3dVector(
    #         np.repeat(cam2.get_center().reshape(1, -1), 3, axis=0).reshape(-1, 3)
    #     )
    # )

    # corr1 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    #     ori1, pred1, [[0, du] for du in range(xyz1.shape[0])]
    # )
    # corr2 = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
    #     ori2, pred2, [[0, du] for du in range(xyz2.shape[0])]
    # )
    # corr1.paint_uniform_color((0.5, 0.5, 0))
    # corr2.paint_uniform_color((1, 0, 0))
    # pred1.paint_uniform_color((0, 0, 1))
    # pred2.paint_uniform_color((1, 0, 0))

    xyz1 = xyz_pred1
    xyz2 = xyz_pred2
    pred1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1))
    pred2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz2))

    pred1.colors = o3d.utility.Vector3dVector(rgb_arr[pid_list1] / 255)
    pred2.colors = o3d.utility.Vector3dVector(rgb_arr[pid_list2] / 255)

    not_inlier1 = np.bitwise_not(np.array(mask1))
    not_inlier2 = np.bitwise_not(np.array(mask2))
    bad_points1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1[not_inlier1]))
    bad_points2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz2[not_inlier2]))
    bad_points1.paint_uniform_color((1, 0, 0))
    bad_points2.paint_uniform_color((1, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=1016)
    parameters = o3d.io.read_pinhole_camera_parameters("viewpoint2.json")
    vis.add_geometry(cam1, reset_bounding_box=True)
    vis.add_geometry(cam3, reset_bounding_box=True)
    vis.add_geometry(cam2, reset_bounding_box=True)
    vis.add_geometry(pred1, reset_bounding_box=True)
    # vis.add_geometry(bad_points1, reset_bounding_box=True)
    vis.add_geometry(pred2, reset_bounding_box=True)
    vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    vis.remove_geometry(cam2, reset_bounding_box=False)
    vis.remove_geometry(pred2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/good-{res_name}.png", do_render=True)
    vis.remove_geometry(cam1, reset_bounding_box=False)
    vis.remove_geometry(pred1, reset_bounding_box=False)
    vis.add_geometry(cam2, reset_bounding_box=False)
    vis.add_geometry(pred2, reset_bounding_box=False)
    # vis.add_geometry(bad_points2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/bad-{res_name}.png", do_render=True)

    # vis.add_geometry(corr2, reset_bounding_box=False)
    # vis.add_geometry(corr1, reset_bounding_box=False)

    # vis.run()
    vis.destroy_window()
    return


def visualize_matches(good_results, bad_results, rgb_arr):
    for idx in range(len(good_results)):
        idx_str = "{:03d}".format(idx)
        make_pic(good_results[idx], bad_results[idx], idx_str, rgb_arr)
    for idx in range(len(good_results)):
        idx_str = "{:03d}".format(idx)
        im1 = cv2.imread(f"debug/good-{idx_str}.png")
        im2 = cv2.imread(f"debug/bad-{idx_str}.png")
        im3 = cv2.hconcat([im2, im1])
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
    ds_name = "Cambridge_GreatCourt"
    print(f"Processing {ds_name}")
    train_ds_ = CambridgeLandmarksDataset(
        train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
    )
    test_ds_ = CambridgeLandmarksDataset(
        train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
    )
    # visualize(train_ds_)

    set1 = set(train_ds_.rgb_files)
    set2 = set(test_ds_.rgb_files)
    set3 = set1.intersection(set2)
    assert len(set3) == 0

    trainer_ = CambridgeLandmarksTrainer(
        train_ds_,
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

    good_name_list = [
        "rgb/seq4_frame00097.png",
        "rgb/seq4_frame00004.png",
        "rgb/seq1_frame00420.png",
        "rgb/seq1_frame00422.png",
        "rgb/seq4_frame00070.png",
    ]

    res = trainer_.process(good_name_list)
    res2 = trainer_2.process(good_name_list)
    visualize_matches(res, res2, trainer_.rgb_arr)

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
        default="../ace/datasets",
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
        default="mixvpr",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=128,
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
