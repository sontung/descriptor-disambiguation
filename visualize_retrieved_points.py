import argparse
import pickle

import numpy as np
import dd_utils
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer
import open3d as o3d
import cv2
import torch
from ace_util import project_using_pose
from tqdm import tqdm


def vconcat_resize_to_smaller(img1, img2):
    """
    Vertically concatenates two images after resizing the larger image to the size of the smaller one.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.

    Returns:
        numpy.ndarray: The concatenated image.
    """
    # Get the shapes of the images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Resize the larger image to the dimensions of the smaller one
    if h1 > h2 or w1 > w2:
        if h1 > h2:
            img1 = cv2.resize(img1, (w1 * h2 // h1, h2))
        if w1 > w2:
            img1 = cv2.resize(img1, (w2, h1 * w2 // w1))
    else:
        if h2 > h1:
            img2 = cv2.resize(img2, (w2 * h1 // h2, h1))
        if w2 > w1:
            img2 = cv2.resize(img2, (w1, h2 * w1 // w2))

    # Concatenate the images vertically
    return cv2.vconcat([img1, img2])


def vconcat_with_pad(img1, img2, pad=10, pad_color=(255, 255, 255)):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    max_width = max(w1, w2)

    # Pad the left side of the narrower image
    # Pad the right side of the narrower image
    if w1 < max_width:
        img1 = cv2.copyMakeBorder(img1, 0, 0, 0, max_width - w1, cv2.BORDER_CONSTANT, value=pad_color)

    if w2 < max_width:
        img2 = cv2.copyMakeBorder(img2, 0, 0, 0, max_width - w2, cv2.BORDER_CONSTANT, value=pad_color)

    # Concatenate using cv2.vconcat
    return cv2.vconcat([img1, img2])


def hconcat_pad(images, pad_value=0):
    # Find the maximum height among images
    max_height = max(img.shape[0] for img in images)

    # Pad each image to have the same height
    padded_images = []
    for img in images:
        h, w, c = img.shape
        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        padded_img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )
        padded_images.append(padded_img)

    # Horizontally concatenate the images
    return cv2.hconcat(padded_images)


def hconcat_resize(images):

    max_height = max(img.shape[0] for img in images)

    # Resize images to the max height while keeping aspect ratio
    resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in images]

    # Concatenate images horizontally
    return cv2.hconcat(resized_images)


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
    return np.mean(np.abs(uv - uv_arr_pred), 1), oob, uv_arr_pred


def make_pic(good_result, bad_result):
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

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1848, height=1016)
    parameters = o3d.io.read_pinhole_camera_parameters("viewpoints_pc_retrieval.json")
    vis.add_geometry(cam1, reset_bounding_box=True)
    vis.add_geometry(cam3, reset_bounding_box=True)
    vis.add_geometry(cam2, reset_bounding_box=True)
    vis.add_geometry(pred1, reset_bounding_box=True)
    vis.add_geometry(pred2, reset_bounding_box=True)
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        parameters, allow_arbitrary=True
    )
    vis.remove_geometry(cam2, reset_bounding_box=False)
    vis.remove_geometry(pred2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/good.png", do_render=True)
    vis.remove_geometry(cam1, reset_bounding_box=False)
    vis.remove_geometry(pred1, reset_bounding_box=False)

    vis.add_geometry(cam2, reset_bounding_box=False)
    vis.add_geometry(pred2, reset_bounding_box=False)
    vis.capture_screen_image(f"debug/bad.png", do_render=True)
    vis.destroy_window()

    return dis1, dis2


def make_cam(pose1):

    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240

    cam1 = o3d.geometry.LineSet.create_camera_visualization(
        427 * 2, 240 * 2, intrinsics, np.vstack([pose1.Rt, [0, 0, 0, 1]]), scale=3
    )

    cam1.paint_uniform_color((0, 0, 0))
    return cam1


def visualize_matches(good_results, bad_results, rgb_arr):
    for idx in tqdm(range(len(good_results))):
        idx_str = "{:03d}".format(idx)
        dis1, dis2 = make_pic(good_results[idx], bad_results[idx], idx_str, rgb_arr)
        score1 = np.sum(dis1<5) / dis1.shape[0]
        score2 = np.sum(dis2<5) / dis2.shape[0]

        im1 = cv2.imread(f"debug/good.png")[150:850, 500:1500]
        im2 = cv2.imread(f"debug/bad.png")[150:, 500:1500]

        font_scale = 2.5
        thick = 5
        # Get the text size to calculate the bottom-right position
        (text_width, text_height), _ = cv2.getTextSize(
            f"{score1*100:.1f}%", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick
        )

        # Calculate the position for the text (bottom-right corner)
        position = (
            im1.shape[1] - text_width - 10,
            text_height + 10,
        )  # 10px margin from the edges
        position2 = (
            im2.shape[1] - text_width - 10,
            text_height + 10,
        )  # 10px margin from the edges

        cv2.putText(
            im1,
            f"{score1*100:.1f}%",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )
        cv2.putText(
            im2,
            f"{score2*100:.1f}%",
            position2,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )
        im3 = cv2.vconcat([im2, im1])
        cv2.imwrite(f"debug/both-{idx_str}.png", im3)

    return


def add_text_above_image(image, text, color=(0, 0, 0)):
    height, width, _ = image.shape
    text_img = np.ones((50, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = text_img.shape[0] // 2 + text_size[1] // 2
    cv2.putText(text_img, text, (text_x, text_y), font, font_scale, color, thickness)
    return cv2.vconcat([text_img, image])


def visualize_matches2(good_results, bad_results, rgb_arr):
    for idx in tqdm(range(len(good_results))):
        assert good_results[idx][0] == bad_results[idx][0]
        idx_str = "{:03d}".format(idx)
        dis1, dis2 = make_pic(good_results[idx], bad_results[idx], idx_str, rgb_arr)

        im1 = cv2.imread(f"debug/good.png")[150:850, 300:1500]
        im2 = cv2.imread(f"debug/bad.png")[150:850, 300:1500]

        if good_results[idx][1]<bad_results[idx][1]:
            color0 = (0, 0, 255)
            color1 = (0, 255, 0)
            tqdm.write(f"idx = {idx}, good = {good_results[idx][1]}, bad = {bad_results[idx][1]}")
        elif good_results[idx][1]>bad_results[idx][1]:
            color0 = (0, 255, 0)
            color1 = (0, 0, 255)
        else:
            color0 = (0, 0, 0)
            color1 = (0, 0, 0)
        im2 = add_text_above_image(im2,
                                   f"Translation error = {bad_results[idx][1]*100:.1f} cm",
                                   color=color1)
        im2 = add_text_above_image(im2, f"Local only")
        im1 = add_text_above_image(im1,
                                   f"Translation error = {good_results[idx][1]*100:.1f} cm",
                                   color=color0)
        im1 = add_text_above_image(im1, f"Local + Global")

        im0 = cv2.imread(f"datasets/cambridge/GreatCourt/{good_results[idx][0]}")
        im0 = cv2.resize(im0, (im0.shape[1] // 7, im0.shape[0] // 7))

        # im3 = hconcat_pad([im0, cv2.hconcat([im1, im2])])
        # cv2.imwrite(f"debug/both-{idx_str}.png", im3)
        h, w,_ = im0.shape
        im2[:h, :w] = im0
        im1[:h, :w] = im0

        cv2.imwrite(f"debug/bad-{idx_str}.png", im2)
        cv2.imwrite(f"debug/both-{idx_str}.png", cv2.hconcat([im2, im1]))
    return


def make_animation(trajectories, pcd):
    parameters = o3d.io.read_pinhole_camera_parameters("viewpoints_pc_retrieval.json")

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1848, height=1016)
    vis.add_geometry(pcd)
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        parameters, allow_arbitrary=True
    )
    for idx in range(len(trajectories[0])):
        for traj in trajectories:
            cam1 = traj[idx]
            vis.add_geometry(cam1, reset_bounding_box=True)
            # vis.get_view_control().convert_from_pinhole_camera_parameters(
            #     parameters, allow_arbitrary=True
            # )
            vis.capture_screen_image(f"debug/ani{idx}.png", do_render=True)
        if idx > 10:
            break
    vis.destroy_window()

    return


def compute_info(good_results, bad_results):
    terrs_good = []
    terrs_bad = []
    rerrs_good = []
    rerrs_bad = []
    for idx00, good_result in enumerate(good_results):
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
        ) = bad_results[idx00]

        gt_pose1 = dd_utils.return_pose_mat_no_inv(gt_pose1.qvec, gt_pose1.tvec)

        xyz1 = xyz_pred1
        xyz2 = xyz_pred2

        dis1, oob1, uv1_pred = compute_reproj_err(gt_pose1, xyz1, uv_arr1)
        dis2, oob2, uv2_pred = compute_reproj_err(gt_pose1, xyz2, uv_arr2)

        score1 = np.sum(oob1) / dis1.shape[0]
        score2 = np.sum(oob2) / dis2.shape[0]
        terrs_good.append(t_err1)
        terrs_bad.append(t_err2)
        rerrs_good.append(score1)
        rerrs_bad.append(score2)
    return terrs_good, terrs_bad, rerrs_good, rerrs_bad


def smooth_measurements(arr, window_size=5, method='gaussian'):

    if method == 'moving_average':
        kernel = np.ones(window_size) / window_size
        return np.convolve(arr, kernel, mode='same')

    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(arr, sigma=window_size / 2)

    elif method == 'median':
        from scipy.ndimage import median_filter
        return median_filter(arr, size=window_size)

    else:
        raise ValueError("Unsupported method. Choose from 'moving_average', 'gaussian', or 'median'.")



def animate_stats(err1, err2, score1, score2):
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulated data length (replace with actual data length)
    num_frames = len(err1)

    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Increased figure size and DPI

    str1 = ("Comparison of Translational Errors and Correct Retrievals\n"
            " (statistics smoothed for visualization purposes)")
    fig.suptitle(str1, fontsize=14)
    # Initialize empty plots
    line_err1, = axs[0].plot([], [], "g", label="local+global")
    line_err2, = axs[0].plot([], [], "r", label="local only")
    axs[0].set_ylabel("Translation error (cm)")
    axs[0].set_xlabel("Frame")
    axs[0].legend()
    axs[0].legend(loc='upper right')  # Place legend at the top-right corner

    line_score1, = axs[1].plot([], [], "g", label="local+global")
    line_score2, = axs[1].plot([], [], "r", label="local only")
    axs[1].set_ylabel("Percentage of correct retrievals (%)")
    axs[1].set_xlabel("Frame")
    axs[1].legend()
    axs[1].legend(loc='lower right')  # Place legend at the top-right corner

    # Iterate over frames and update manually
    for frame in range(num_frames):
        x_vals = np.arange(frame + 1)  # Frames up to current frame

        line_err1.set_data(x_vals, err1[:frame + 1])
        line_err2.set_data(x_vals, err2[:frame + 1])
        line_score1.set_data(x_vals, score1[:frame + 1])
        line_score2.set_data(x_vals, score2[:frame + 1])

        axs[0].set_xlim(0, num_frames)
        axs[1].set_xlim(0, num_frames)
        axs[0].set_ylim(0, max(max(err1), max(err2)) * 1.1)
        axs[1].set_ylim(0, max(max(score1), max(score2)) * 1.1)

        plt.tight_layout()
        plt.savefig(f"debug/stats_{frame:03d}.png", dpi=150)  # High DPI for better quality
        # break


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
        lambda_val=0.5,
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
    # visualize_matches(res, res_bad, trainer_.rgb_arr)

    res00 = sorted([du for du in res if "seq1" in du[0]], key=lambda du: du[0])
    res10 = sorted([du for du in res_bad if "seq1" in du[0]], key=lambda du: du[0])
    visualize_matches2(res00, res10, trainer_.rgb_arr)

    err1, err2, score1, score2 = compute_info(res00, res10)
    err1 = smooth_measurements(err1, window_size=10)*10
    err2 = smooth_measurements(err2, window_size=10)*10
    score1 = smooth_measurements(score1, window_size=10)
    score2 = smooth_measurements(score2, window_size=10)

    animate_stats(err1, err2, score1, score2)

    for i in range(612):
        id_ = f"{i:03d}"
        im1 = cv2.imread(f"debug/both-{id_}.png")
        im2 = cv2.imread(f"debug/stats_{id_}.png")
        # im2 = cv2.resize(im2, (im2.shape[1]*2//3, im2.shape[0]*2//3))
        im = vconcat_with_pad(im2, im1)
        cv2.imwrite(f"debug/final-{id_}.png", im)
        # break

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
