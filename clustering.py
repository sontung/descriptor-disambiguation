import random

import numpy as np
import faiss
import open3d as o3d
from numba.core.serialize import pickle
from pykdtree.kdtree import KDTree
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from dataset import CambridgeLandmarksDataset


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


def main2():
    file_name1 = f"output/GreatCourt/image_desc_eigenplaces_ResNet101_2048_2048.npy"
    file_name2 = (
        f"output/GreatCourt/image_desc_name_eigenplaces_ResNet101_2048_2048.npy"
    )
    all_desc = np.load(file_name1)
    afile = open(file_name2, "rb")
    all_names = pickle.load(afile)
    afile.close()
    image2desc = {}
    for idx, name in enumerate(all_names):
        image2desc[name] = all_desc[idx]
    niter = 20
    verbose = True
    n_centroids = 100
    d = all_desc.shape[1]
    kmeans = faiss.Kmeans(d, n_centroids, niter=niter, verbose=verbose)
    kmeans.train(all_desc)
    _, indices = kmeans.index.search(all_desc, 1)
    indices = indices.flatten()

    train_ds_ = CambridgeLandmarksDataset(
        train=True, ds_name="GreatCourt", root_dir="datasets/cambridge"
    )

    cluster_ind = 0
    mask = indices == cluster_ind
    names_curr = np.array(all_names)[mask]
    image_ids_curr = [
        train_ds_.image_name2id[name.split(train_ds_.images_dir)[-1][1:]]
        for name in names_curr
    ]
    pid_curr = [train_ds_.image_id2pids[img_id] for img_id in image_ids_curr]
    pid_curr, counts = np.unique(np.concatenate(pid_curr), return_counts=True)
    # pid_curr = pid_curr[counts>10]

    xyz_arr = np.array([train_ds_.recon_points[pid].xyz for pid in pid_curr])

    clustering = DBSCAN(eps=1, min_samples=10).fit(xyz_arr)
    colors = np.random.random((np.max(clustering.labels_) + 1, 3))
    colors[-1] = [0, 0, 0]
    mask2 = clustering.labels_ != -1
    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[mask2]))
    pc1.colors = o3d.utility.Vector3dVector(colors[clustering.labels_][mask2])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc1)
    vis.run()
    vis.destroy_window()

    centroids = xyz_arr[mask2]
    votes = np.zeros(centroids.shape[0], int)
    for img_id in image_ids_curr:
        xyz_img = np.array(
            [train_ds_.recon_points[pid].xyz for pid in train_ds_.image_id2pids[img_id]]
        )
        tree = KDTree(centroids)
        dis, ind = tree.query(xyz_img)
        mask3 = dis < 1
        votes[ind[mask3]] += 1

    mask3 = votes >= len(image_ids_curr) // 2
    print(np.sum(mask3), np.max(votes), len(image_ids_curr))
    colors = np.zeros_like(centroids)
    colors[mask3] = [1, 0, 0]
    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroids))
    pc1.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc1)
    vis.run()
    vis.destroy_window()

    print()


# @profile
def reduce_map_using_min_cover(train_ds_, vis=False, min_cover=100):
    # train_ds_ = CambridgeLandmarksDataset(
    #     train=True, ds_name="GreatCourt", root_dir="datasets/cambridge"
    # )
    pid2images = {
        pid: list(train_ds_.recon_points[pid].image_ids)
        for pid in train_ds_.recon_points
    }
    pid2ind = {}
    ind2pid = {}
    score_mat = np.zeros(len(train_ds_.recon_points), int)
    available_mat = np.ones(len(train_ds_.recon_points), bool)
    index_arr = np.arange(len(train_ds_.recon_points))
    for ind, pid in enumerate(train_ds_.recon_points.keys()):
        pid2ind[pid] = ind
        ind2pid[ind] = pid
        score_mat[ind] = len(pid2images[pid])

    image2covers = {img_id: 0 for img_id in train_ds_.recon_images}
    image2indices = {
        img_id: [pid2ind[pid] for pid in train_ds_.image_id2pids[img_id]]
        for img_id in train_ds_.recon_images
    }

    pid_list = set(list(pid2images.keys()))
    chosen_pid = set([])
    ori_size = len(pid_list)
    done_images = set([])

    for image in train_ds_.image_id2pids:
        pid_list_curr = train_ds_.image_id2pids[image]
        indices = image2indices[image]
        if len(pid_list_curr) < min_cover:
            chosen_pid.update(pid_list_curr)
            done_images.add(image)
            available_mat[indices] = False

    pbar = tqdm(total=len(image2covers) - len(done_images), desc="Covering images")

    while True:
        best_score = np.max(score_mat[available_mat])
        mask = np.bitwise_and(available_mat, score_mat == best_score)
        best_indices = index_arr[mask]

        if best_score == 0:
            for image in image2covers:
                if image not in done_images:
                    pid_list_curr = train_ds_.image_id2pids[image]
                    for pid in pid_list_curr:
                        if pid not in chosen_pid:
                            chosen_pid.add(pid)
                            available_mat[pid2ind[pid]] = False
                            image2covers[image] += 1
                            if image2covers[image] >= min_cover:
                                break
            break
        best_index = random.choice(best_indices)
        best_pid = ind2pid[best_index]
        images_covered_list = []
        for image in pid2images[best_pid]:
            image2covers[image] += 1
            if image2covers[image] >= min_cover and image not in done_images:
                done_images.add(image)
                images_covered_list.append(image)
                pbar.update(1)

        for image in images_covered_list:
            indices = image2indices[image]
            score_mat[indices] -= 1

        assert best_pid not in chosen_pid
        chosen_pid.add(best_pid)
        pid_list.remove(best_pid)
        available_mat[pid2ind[best_pid]] = False

        if len(done_images) == len(image2covers):
            break

    pbar.close()
    for image in train_ds_.image_id2pids:
        pid_list = train_ds_.image_id2pids[image]
        if len(pid_list) >= min_cover:
            res = len([pid for pid in pid_list if pid in chosen_pid])
            assert res >= min_cover, image
    print(len(chosen_pid) / ori_size)

    if vis:
        xyz_arr = np.array([train_ds_.recon_points[pid].xyz for pid in chosen_pid])

        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr))
        pc1, inlier_ind = pc1.remove_radius_outlier(nb_points=16, radius=5)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc1)
        vis.run()
        vis.destroy_window()

    train_ds_.image_id2pids = {}
    train_ds_.image_id2uvs = {}
    for img_id in tqdm(train_ds_.recon_images, desc="Gathering points per image"):
        pid_arr = train_ds_.recon_images[img_id].point3D_ids
        mask = [True if pid in chosen_pid else False for pid in pid_arr]
        train_ds_.image_id2pids[img_id] = pid_arr[mask]
        train_ds_.image_id2uvs[img_id] = train_ds_.recon_images[img_id].xys[mask]

    pid_list = list(train_ds_.recon_points.keys())
    for pid in pid_list:
        if pid not in chosen_pid:
            del train_ds_.recon_points[pid]


if __name__ == "__main__":
    reduce_map_using_min_cover()
