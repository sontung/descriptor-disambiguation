import os.path
import pickle
import random

import open3d as o3d
import numpy as np

import dd_utils
from dataset import CambridgeLandmarksDataset
from sklearn.decomposition import PCA

from trainer import CambridgeLandmarksTrainer
from matplotlib.colors import hsv_to_rgb


def load_desc(file_name1, file_name2, dataset):
    pid2mean_desc = np.load(file_name1)
    all_pid = np.load(file_name2)
    xyz_arr = dataset.xyz_arr[all_pid]

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
    cl, inlier_ind = point_cloud.remove_radius_outlier(
        nb_points=16, radius=5, print_progress=True
    )
    colors = process_colors(pid2mean_desc[inlier_ind])
    cl.colors = o3d.utility.Vector3dVector(colors)

    return cl


def process_colors(desc):
    pca = PCA(whiten=False, n_components=3)
    colors_ori = pca.fit_transform(desc)
    min_arr = np.min(colors_ori, 1).reshape(-1, 1)
    max_arr = np.max(colors_ori, 1).reshape(-1, 1)
    # min_arr = np.min(colors_ori, 0)
    # max_arr = np.max(colors_ori, 0)
    colors = (colors_ori - min_arr) / (max_arr - min_arr)
    return colors


def render_images(cl, trainer_):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cl)

    for idx0 in range(3):
        parameters = o3d.io.read_pinhole_camera_parameters(f"viewpoint{idx0}.json")
        vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
        vis.capture_screen_image(f"debug/test-{idx0}-{trainer_.lambda_val}.png", do_render=True)
    vis.run()
    vis.destroy_window()


def main():
    train_ds_ = CambridgeLandmarksDataset(
        train=True, ds_name="GreatCourt", root_dir="datasets/cambridge"
    )
    test_ds_ = CambridgeLandmarksDataset(
        train=False, ds_name="GreatCourt", root_dir="datasets/cambridge"
    )
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        "d2net", "mixvpr", 512
    )

    trainer_ = CambridgeLandmarksTrainer(
        train_ds_,
        test_ds_,
        512,
        512,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        True,
        lambda_val=0.5,
        convert_to_db_desc=False,
    )

    nb_regions = 4

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(trainer_.xyz_arr))
    cl, inlier_ind = point_cloud.remove_radius_outlier(
        nb_points=512, radius=5, print_progress=True
    )

    if os.path.isfile("colors.npy"):
        colors = np.load("colors.npy")
    else:
        hues = np.linspace(0, 1, nb_regions, endpoint=False)  # Equally spaced hues
        random.shuffle(hues)
        hsv_colors = np.array([[hue,
                                random.randint(70, 100)/100,
                                random.randint(70, 100)/100] for hue in hues])
        colors = hsv_to_rgb(hsv_colors)
        np.save("colors.npy", colors)

    indices, _ = dd_utils.cluster_by_faiss_kmeans(trainer_.pid2mean_desc, nb_regions)
    cl.colors = o3d.utility.Vector3dVector(colors[indices[inlier_ind]])
    render_images(cl, trainer_)

    # trainer_.lambda_val = 0
    # trainer_.pid2mean_desc = trainer_.collect_descriptors()
    # indices, _ = dd_utils.cluster_by_faiss_kmeans(trainer_.pid2mean_desc, nb_regions)
    # cl.colors = o3d.utility.Vector3dVector(colors[indices[inlier_ind]])
    # render_images(cl, trainer_)

    trainer_.lambda_val = 1
    trainer_.pid2mean_desc = trainer_.collect_descriptors()
    indices, _ = dd_utils.cluster_by_faiss_kmeans(trainer_.pid2mean_desc, nb_regions)
    cl.colors = o3d.utility.Vector3dVector(colors[indices[inlier_ind]])
    render_images(cl, trainer_)

    # vis = o3d.visualization.Visualizer()
    # cl.colors = o3d.utility.Vector3dVector(colors[indices[inlier_ind]])
    # vis.create_window()
    # vis.add_geometry(cl)
    #
    # vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("viewpoint2.json", param)
    # vis.destroy_window()
    return


def load_desc_robot_car(file_name1, file_name2):
    pid2mean_desc = np.load(file_name1)
    xyz_arr = np.load(file_name2)

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
    colors = process_colors(pid2mean_desc)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def save_all_colors():
    file_name1 = f"output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-0.npy"
    pid2mean_desc = np.load(file_name1)
    colors = process_colors(pid2mean_desc)
    np.save(f"output/robotcar/colors0.npy", colors)
    file_name1 = f"output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-1.npy"
    pid2mean_desc = np.load(file_name1)
    colors = process_colors(pid2mean_desc)
    np.save(f"output/robotcar/colors1.npy", colors)
    file_name1 = (
        f"output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-0.5.npy"
    )
    pid2mean_desc = np.load(file_name1)
    colors = process_colors(pid2mean_desc)
    np.save(f"output/robotcar/colors0.5.npy", colors)


def main_robot_car():
    file_name1 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-0.npy"
    file_name2 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/xyz_arrd2net-eigenplaces_ResNet101_2048-0.npy"

    point_cloud1 = load_desc_robot_car(file_name1, file_name2)

    file_name1 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-1.npy"
    file_name2 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/xyz_arrd2net-eigenplaces_ResNet101_2048-1.npy"

    point_cloud2 = load_desc_robot_car(file_name1, file_name2)
    point_cloud2.translate([250, 0, 0])

    file_name1 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_descd2net-eigenplaces_ResNet101_2048-0.5.npy"
    file_name2 = f"/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/xyz_arrd2net-eigenplaces_ResNet101_2048-0.5.npy"

    point_cloud3 = load_desc_robot_car(file_name1, file_name2)
    point_cloud3.translate([0, 250, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # ctr = vis.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters("viewpoint.json")

    vis.add_geometry(point_cloud1)
    vis.add_geometry(point_cloud2)
    vis.add_geometry(point_cloud3)
    # vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)

    # vis.capture_screen_image("debug/test3.png", do_render=True)

    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)
    vis.destroy_window()
    return


if __name__ == "__main__":
    main()
    # save_all_colors()
    # main_robot_car()
