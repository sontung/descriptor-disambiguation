import numpy as np
import faiss
import open3d as o3d
from pykdtree.kdtree import KDTree
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def cluster(x, nb_clusters):
    niter = 20
    d = x.shape[1]

    kmeans = faiss.Kmeans(d, int(nb_clusters), niter=niter, verbose=False)
    kmeans.train(x)

    _, indices = kmeans.index.search(x, 1)

    indices = indices.flatten()
    return indices


def cluster_function(desc, xyz_arr):
    niter = 20
    verbose = True
    n_centroids = 20000
    d = desc.shape[1]
    kmeans = faiss.Kmeans(d, n_centroids, niter=niter, verbose=verbose)
    kmeans.train(desc)
    _, indices = kmeans.index.search(desc, 1)

    indices = indices.flatten()

    cid_list, counts = np.unique(indices, return_counts=True)
    keep = np.zeros(xyz_arr.shape[0], dtype=int)
    point_indices = np.arange(xyz_arr.shape[0])
    for idx, current_cid in enumerate(tqdm(cid_list)):
        count = counts[idx]
        mask = indices == current_cid
        current_xyz = xyz_arr[mask]
        current_point_indices = point_indices[mask]
        if count >= 2:
            clustering = DBSCAN(eps=1, min_samples=2).fit(current_xyz)
            keep[current_point_indices[clustering.labels_ == -1]] = 0
            if np.max(clustering.labels_) > 0:
                keep[current_point_indices] = 0
                continue
            for i in range(np.max(clustering.labels_)):
                indices_in_question = current_point_indices[clustering.labels_ == i]
                keep[indices_in_question] = 1
                # desc[indices_in_question[0]] = np.mean(desc[indices_in_question], 0)
        else:
            keep[current_point_indices] = 1

    print(np.sum(keep) / keep.shape[0])
    mask = keep.astype(bool)
    # np.save("output/GreatCourt/codebook_mask.npy", point_indices[keep.astype(bool)])
    return desc[point_indices[mask]], xyz_arr[point_indices[mask]]


def main():
    desc = np.load("output/GreatCourt/codebook-d2net-mixvpr_512.npy")
    xyz_arr = np.load("output/GreatCourt/xyz-d2net-mixvpr_512.npy")

    # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
    # cl, inlier_ind = point_cloud.remove_radius_outlier(
    #     nb_points=16, radius=5, print_progress=True
    # )
    # desc = desc[inlier_ind]
    # xyz_arr = xyz_arr[inlier_ind]
    niter = 20
    verbose = True
    n_centroids = 20000
    d = desc.shape[1]
    kmeans = faiss.Kmeans(d, n_centroids, niter=niter, verbose=verbose)
    kmeans.train(desc)
    _, indices = kmeans.index.search(desc, 1)

    indices = indices.flatten()

    cid_list, counts = np.unique(indices, return_counts=True)
    keep = np.zeros(xyz_arr.shape[0], dtype=int)
    point_indices = np.arange(xyz_arr.shape[0])
    colors = np.zeros(xyz_arr.shape[0], dtype=int)
    color_idx = -1
    for idx, current_cid in enumerate(tqdm(cid_list)):
        count = counts[idx]
        mask = indices == current_cid
        current_xyz = xyz_arr[mask]
        current_point_indices = point_indices[mask]
        if count >= 2:
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(current_xyz)
            keep[current_point_indices[clustering.labels_ == -1]] = 0
            for pid in current_point_indices[clustering.labels_ == -1]:
                color_idx += 1
                colors[pid] = color_idx
            if np.max(clustering.labels_) > 0:
                keep[current_point_indices] = 0
                continue
            for i in range(np.max(clustering.labels_)):
                indices_in_question = current_point_indices[clustering.labels_ == i]
                keep[indices_in_question] = 1
                desc[indices_in_question[0]] = np.mean(desc[indices_in_question], 0)
                color_idx += 1
                colors[indices_in_question] = color_idx
        else:
            keep[current_point_indices] = 1

    print(np.sum(keep) / keep.shape[0])
    mask = keep.astype(bool)
    colors = np.random.random((np.max(colors) + 1, 3))

    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[mask]))
    pc1.colors = o3d.utility.Vector3dVector(colors[indices[mask]])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc1)
    vis.run()
    vis.destroy_window()
    np.save("output/GreatCourt/codebook_mask.npy", point_indices[keep.astype(bool)])


if __name__ == "__main__":
    main()
