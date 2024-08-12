import pickle
import cv2
import faiss
import h5py
import numpy as np
from pykdtree.kdtree import KDTree

import dd_utils
from tqdm import tqdm

from dataset import RobotCarDataset


def perform_clustering(name, nb_clusters=1000):
    arr = np.load(name)
    niter = 20
    d = arr.shape[1]

    kmeans = faiss.Kmeans(d, int(nb_clusters), niter=niter, verbose=True, gpu=True)
    kmeans.train(arr)

    _, indices = kmeans.index.search(arr, 1)

    indices = indices.flatten()
    return indices, kmeans.index


def compute_acc(pred, gt):
    diff = pred - gt
    acc = np.sum(diff == 0) / diff.shape[0]
    return acc, diff


def match(xq, xb):
    index = faiss.IndexFlatL2(xq.shape[1])  # build the index
    index.add(xb)
    dis, ind = index.search(xq, 1)
    print(ind)
    return ind


def get_index(xb):
    index = faiss.IndexFlatL2(xb.shape[1])  # build the index
    index.add(xb)
    return index


def main():
    afile = open(
        "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2ind-d2net-salad_8448.npy",
        "rb",
    )
    pid2ind = pickle.load(afile)
    afile.close()
    desc_global = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/codebook-d2net-salad_8448.npy"
    all_desc = np.load(desc_global)

    indices, _ = dd_utils.cluster_by_faiss_kmeans(all_desc, 1000, True)
    all_clusters, counts = np.unique(indices, return_counts=True)
    cluster_indices_sorted = np.argsort(counts)

    ds = RobotCarDataset()

    pid2im = {pid: [] for pid in pid2ind}
    im2pid = {}
    for img_id in ds.image2points:
        for pid in ds.image2points[img_id]:
            if pid in pid2im:
                pid2im[pid].append(img_id)
                im2pid.setdefault(img_id, []).append(pid)
    point_indices = np.arange(all_desc.shape[0])
    point_pids = np.zeros(all_desc.shape[0], int)
    for pid, ind in pid2ind.items():
        point_pids[ind] = pid

    im_fills = {img_id: 0 for img_id in im2pid}
    selected_clusters = []
    for cluster_idx in tqdm(cluster_indices_sorted):
        cluster = all_clusters[cluster_idx]
        selected_clusters.append(cluster)
        mask = indices==cluster
        point_ind = point_indices[mask]
        point_pid = point_pids[point_ind]
        for pid in point_pid:
            for im in pid2im[pid]:
                if im in im_fills:
                    im_fills[im] += 1
        bad = [im for im in im_fills if im_fills[im] < 20]
        # map_filled = np.sum(np.isin(indices, selected_clusters))/all_desc.shape[0]
        db_images_filled = 1-len(bad)/len(im_fills)
        if db_images_filled > 0.9:
            break
    print(len(selected_clusters)/all_clusters.shape[0])

    mask = np.isin(indices, selected_clusters)
    xyz_arr = np.zeros((all_desc.shape[0], 3))
    for pid in pid2ind:
        xyz_arr[pid2ind[pid]] = ds.xyz_arr[pid]

    np.save("output/robotcar/good_pids.npy", point_pids[mask])

    # import open3d as o3d
    # pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[np.bitwise_not(mask)]))
    # cl1, _ = pc1.remove_radius_outlier(
    #     nb_points=16, radius=5, print_progress=True
    # )
    # pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
    # cl2, _ = pc2.remove_radius_outlier(
    #     nb_points=16, radius=5, print_progress=True
    # )
    # cl1.paint_uniform_color((0, 1, 0))
    # cl2.paint_uniform_color((1, 0, 0))
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry(cl2)
    # vis.add_geometry(cl1)
    # vis.run()
    # vis.destroy_window()

    return


if __name__ == "__main__":
    main()
