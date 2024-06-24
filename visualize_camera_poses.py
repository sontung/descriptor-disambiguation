import pickle

import torch

import dd_utils
import numpy as np
import open3d as o3d
from dataset import RobotCarDataset
from tqdm import tqdm


def main():
    desc_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/image_desc_salad_8448.npy"
    name_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/image_desc_name_salad_8448.npy"
    train_ds_ = RobotCarDataset(ds_dir="datasets/robotcar")
    all_desc = np.load(desc_file)
    afile = open(name_file, "rb")
    all_names = pickle.load(afile)
    afile.close()

    nb_clusters = 100
    cluster_indices, _ = dd_utils.cluster_by_faiss_kmeans(all_desc, nb_clusters)
    colors = np.random.random((nb_clusters, 3))
    color_arr = colors[cluster_indices]
    name2color = {name.split("images/")[-1]: cluster_indices[idx] for idx, name in enumerate(all_names)}

    intrinsics = np.eye(3)

    intrinsics[0, 0] = 738
    intrinsics[1, 1] = 738
    intrinsics[0, 2] = 427  # 427
    intrinsics[1, 2] = 240

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 0.5
    # vis.add_geometry(pc1)

    all_poses = torch.zeros((len(train_ds_), 4, 4))
    all_cluster_ids = []
    for idx in tqdm(range(0, len(train_ds_))):
        example = train_ds_[idx]
        pose_mat = example[4]
        all_poses[idx] = pose_mat
        cluster_id = name2color[example[1].split("images/")[-1]]
        all_cluster_ids.append(cluster_id)
        # if cluster_id == 0:
        #     cam = o3d.geometry.LineSet.create_camera_visualization(
        #         427 * 2, 240 * 2, intrinsics, pose_mat, scale=10
        #     )
        #     vis.add_geometry(cam)
        #     cam.paint_uniform_color(colors[cluster_id])
    vis.run()
    vis.destroy_window()

    for cluster_id in range(nb_clusters):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1848, height=1016)
        mask = np.array(all_cluster_ids)==cluster_id
        poses = all_poses[mask]
        for pose in poses:
            cam = o3d.geometry.LineSet.create_camera_visualization(
                427 * 2, 240 * 2, intrinsics, pose, scale=10
            )
            cam.paint_uniform_color(colors[0])
            vis.add_geometry(cam)
        vis.capture_screen_image(
            f"debug/test{cluster_id}.png",
            do_render=True,
        )
        vis.destroy_window()

    return


if __name__ == '__main__':
    main()
