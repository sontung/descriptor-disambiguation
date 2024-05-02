import pickle
import open3d as o3d
import numpy as np
from dataset import CambridgeLandmarksDataset
from sklearn.decomposition import PCA


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


def main():
    train_ds_ = CambridgeLandmarksDataset(
        train=True,
        ds_name="Cambridge_GreatCourt",
        root_dir="../ace/datasets/Cambridge_GreatCourt",
    )

    file_name1 = f"output/Cambridge_GreatCourt/codebook_r2d2.npy"
    file_name2 = f"output/Cambridge_GreatCourt/all_pids_r2d2.npy"

    point_cloud1 = load_desc(file_name1, file_name2, train_ds_)

    file_name1 = f"output/Cambridge_GreatCourt/codebook_r2d2_mixvpr_128.npy"
    file_name2 = f"output/Cambridge_GreatCourt/all_pids_r2d2_mixvpr_128.npy"

    point_cloud2 = load_desc(file_name1, file_name2, train_ds_)
    # point_cloud2.translate([250, 0, 0])

    file_name1 = f"output/Cambridge_GreatCourt/codebook_r2d2_mixvpr_128_0.npy"
    file_name2 = f"output/Cambridge_GreatCourt/all_pids_r2d2_mixvpr_128_0.npy"

    point_cloud3 = load_desc(file_name1, file_name2, train_ds_)
    # point_cloud3.translate([0, 250, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("viewpoint.json")

    # vis.add_geometry(point_cloud1)
    # vis.add_geometry(point_cloud2)
    vis.add_geometry(point_cloud3)
    vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)

    vis.capture_screen_image("debug/test3.png", do_render=True)

    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)
    vis.destroy_window()
    return


if __name__ == "__main__":
    main()
