import random
import sys

import h5py
import numpy as np
import faiss

# import open3d as o3d
import pickle
from pykdtree.kdtree import KDTree
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import dd_utils
from dataset import CambridgeLandmarksDataset, RobotCarDataset
from trainer import retrieve_pid


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


def collect_codebook(dataset):
    features_h5 = h5py.File("output/GreatCourt/d2net_features_train.h5", "r")

    pid2mean_desc = np.zeros(
        (len(dataset.recon_points), 512),
    )
    pid2count = np.zeros(len(dataset.recon_points))
    index_for_array = -1
    pid2ind = {}
    for example in tqdm(dataset, desc="Collecting point descriptors"):
        if example is None:
            continue

        keypoints, descriptors = dd_utils.read_kp_and_desc(example[1], features_h5)

        pid_list = example[3]
        uv = example[-1]

        selected_pid, mask, ind = retrieve_pid(pid_list, uv, keypoints)
        selected_descriptors = descriptors[ind[mask]]
        for idx, pid in enumerate(selected_pid):
            if pid not in pid2ind:
                index_for_array += 1
                pid2ind[pid] = index_for_array
        dataset.image_id2pids[example[2]] = np.unique(selected_pid)
        idx2 = [pid2ind[pid] for pid in selected_pid]
        pid2mean_desc[idx2] += selected_descriptors
        pid2count[idx2] += 1
    index_for_array += 1
    pid2mean_desc = pid2mean_desc[:index_for_array, :] / pid2count[
        :index_for_array
    ].reshape(-1, 1)
    features_h5.close()

    niter = 20
    verbose = True
    n_centroids = 1000
    d = pid2mean_desc.shape[1]
    kmeans = faiss.Kmeans(d, n_centroids, niter=niter, verbose=verbose)
    kmeans.train(pid2mean_desc)
    _, indices = kmeans.index.search(pid2mean_desc, 1)
    indices = indices.flatten()
    pid2cid = {pid: indices[pid2ind[pid]] for pid in pid2ind}

    return pid2cid


def cluster_into_hyperpoints(
    train_ds_, pid2ind, ind2pid, score_mat, pid2cid, xyz_arr, pid_arr, vis=False
):
    index_arr = np.arange(len(train_ds_.recon_points))
    ind2cluster = np.zeros(len(train_ds_.recon_points), int) - 1
    available_mat = np.ones(len(train_ds_.recon_points), bool)
    invalid = [pid2ind[pid] for pid in pid_arr if pid not in pid2cid]
    available_mat[invalid] = False
    cluster_ind_0 = 0
    tree = BallTree(xyz_arr)
    while True:
        if np.sum(available_mat) <= 1:
            break
        best_score = np.max(score_mat[available_mat])
        print(best_score, np.sum(available_mat))
        mask = np.bitwise_and(available_mat, score_mat == best_score)
        best_indices = index_arr[mask]
        for best_index in best_indices:
            best_pid = ind2pid[best_index]
            cid0 = pid2cid[best_pid]
            # dis_3d = np.mean(np.abs(xyz_arr[best_index] - xyz_arr), 1)
            # mask = np.bitwise_and(dis_3d < 1, available_mat)
            res = tree.query_radius(
                xyz_arr[best_index].reshape(-1, 3), r=1, return_distance=True
            )
            res = res[0][0][res[1][0] > 0]
            res = res[available_mat[res]]
            pid_list = [ind2pid[ind] for ind in res]
            list_0 = [best_pid]
            for pid in pid_list:
                if pid not in list_0 and pid in pid2cid and pid2cid[pid] == cid0:
                    list_0.append(pid)
            indices = [pid2ind[pid] for pid in list_0]
            available_mat[indices] = False
            if len(list_0) == 1:
                continue
            ind2cluster[indices] = cluster_ind_0
            cluster_ind_0 += 1
    if vis:
        colors = np.random.random((np.max(ind2cluster) + 1, 3))
        colors[-1] = [0, 0, 0]
        clusters, counts = np.unique(ind2cluster, return_counts=True)
        mask2 = np.bitwise_and(
            np.isin(ind2cluster, clusters[counts > 2]), ind2cluster != -1
        )
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[mask2]))
        pc1.colors = o3d.utility.Vector3dVector(colors[ind2cluster][mask2])
        cl, inlier_ind = pc1.remove_radius_outlier(nb_points=16, radius=5)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc1)
        vis.run()
        vis.destroy_window()
    return ind2cluster


# @profile
def reduce_map_using_min_cover(train_ds_, vis=False, min_cover=100):
    # train_ds_ = CambridgeLandmarksDataset(
    #     train=True, ds_name="GreatCourt", root_dir="datasets/cambridge"
    # )

    xyz_arr = np.zeros((len(train_ds_.recon_points), 3))
    pid_arr = np.zeros(len(train_ds_.recon_points), int)

    pid2images = {
        pid: list(train_ds_.recon_points[pid].image_ids)
        for pid in train_ds_.recon_points
    }
    pid2ind = {}
    ind2pid = {}
    score_mat = np.zeros(len(train_ds_.recon_points), int)
    index_arr = np.arange(len(train_ds_.recon_points))
    for ind, pid in enumerate(train_ds_.recon_points.keys()):
        xyz_arr[ind] = train_ds_.recon_points[pid].xyz
        pid_arr[ind] = pid
        pid2ind[pid] = ind
        ind2pid[ind] = pid
        score_mat[ind] = len(pid2images[pid])

    pid2cid = collect_codebook(train_ds_)
    ind2cluster = cluster_into_hyperpoints(
        train_ds_, pid2ind, ind2pid, score_mat, pid2cid, xyz_arr, pid_arr, vis
    )
    cluster2images = {}
    cluster2pids = {}
    for cluster_id in range(np.max(ind2cluster) + 1):
        mask = ind2cluster == cluster_id
        indices = index_arr[mask]
        pid_list = [ind2pid[ind] for ind in indices]
        images = []
        for pid in pid_list:
            images.extend(pid2images[pid])
        cluster2images[cluster_id] = set(images)
        cluster2pids[cluster_id] = pid_list

    image2pid = {}
    for cluster_id in cluster2images:
        for image in cluster2images[cluster_id]:
            image2pid.setdefault(image, []).append(cluster_id)
    image2covers = {img_id: 0 for img_id in image2pid}

    chosen_cluster = set([])
    done_images = set([])
    available_mat = np.ones(np.max(ind2cluster) + 1, bool)
    score_mat = np.zeros(np.max(ind2cluster) + 1, int)
    index_arr = np.arange(np.max(ind2cluster) + 1)
    for cluster_id in cluster2images:
        score_mat[cluster_id] = len(cluster2images[cluster_id])

    for image in image2pid:
        indices = image2pid[image]
        if len(indices) < min_cover:
            chosen_cluster.update(indices)
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
                    cluster_list = image2pid[image]
                    for cluster_id in cluster_list:
                        if cluster_id not in chosen_cluster:
                            chosen_cluster.add(cluster_id)
                            available_mat[cluster_id] = False
                            image2covers[image] += 1
                            if image2covers[image] >= min_cover:
                                break
            break
        best_cluster_id = random.choice(best_indices)
        images_covered_list = []
        for image in cluster2images[best_cluster_id]:
            image2covers[image] += 1
            if image2covers[image] >= min_cover and image not in done_images:
                done_images.add(image)
                images_covered_list.append(image)
                pbar.update(1)

        for image in images_covered_list:
            indices = image2pid[image]
            score_mat[indices] -= 1

        assert best_cluster_id not in chosen_cluster
        chosen_cluster.add(best_cluster_id)
        available_mat[best_cluster_id] = False

        if len(done_images) == len(image2covers):
            break

    pbar.close()
    # for image in image2pid:
    #     pid_list = image2pid[image]
    #     if len(pid_list) >= min_cover:
    #         res = len([pid for pid in pid_list if pid in chosen_cluster])
    #         assert res >= min_cover, image

    print(len(chosen_cluster) / len(cluster2images))
    mask = np.isin(ind2cluster, list(chosen_cluster))
    colors = np.zeros((np.max(ind2cluster) + 2, 3))
    colors[list(chosen_cluster)] = np.random.random((len(chosen_cluster), 3))
    print(np.sum(mask) / xyz_arr.shape[0])
    chosen_pid = set(list(pid_arr[mask]))
    if vis:
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_arr[mask]))
        # pc1, inlier_ind = pc1.remove_radius_outlier(nb_points=16, radius=5)
        pc1.colors = o3d.utility.Vector3dVector(colors[ind2cluster[mask]])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc1)
        vis.run()
        vis.destroy_window()

    # train_ds_.image_id2pids = {}
    # train_ds_.image_id2uvs = {}
    # for img_id in tqdm(train_ds_.recon_images, desc="Gathering points per image"):
    #     curr_pid_arr = train_ds_.recon_images[img_id].point3D_ids
    #     mask = [True if pid in chosen_pid else False for pid in curr_pid_arr]
    #     train_ds_.image_id2pids[img_id] = curr_pid_arr[mask]
    #     train_ds_.image_id2uvs[img_id] = train_ds_.recon_images[img_id].xys[mask]
    #
    # pid_list = list(train_ds_.recon_points.keys())
    # for pid in pid_list:
    #     if pid not in chosen_pid:
    #         del train_ds_.recon_points[pid]
    return chosen_pid


def reduce_visible_set(train_ds_, all_desc, all_names):
    indices, _ = dd_utils.cluster_by_faiss_kmeans(all_desc, 500, True)
    img2cid = {all_names[i]: indices[i] for i in range(len(indices))}

    pid2images = {}
    for img in train_ds_.image2points:
        name = train_ds_.image2name[img]
        name2 = f"{train_ds_.images_dir}/{name[2:].replace('png', 'jpg')}"
        cid = img2cid[name2]
        for pid in train_ds_.image2points[img]:
            pid2images.setdefault(pid, []).append(cid)

    for pid in pid2images.keys():
        pid2images[pid] = "-".join(map(str, sorted(set(pid2images[pid]))))

    image2pid = {}
    for pid in pid2images:
        image = pid2images[pid]
        image2pid.setdefault(image, []).append(pid)

    final_desc = np.zeros((len(image2pid), all_desc.shape[1]))
    count = 0
    image2count = {}
    for image in image2pid:
        cid_list = set(map(int, image.split("-")))
        mask = [True if ind in cid_list else False for ind in indices]
        desc = all_desc[mask]
        desc_m = np.mean(desc, 0)
        final_desc[count] = desc_m
        image2count[image] = count
        count += 1

    for pid in pid2images.keys():
        pid2images[pid] = image2count[pid2images[pid]]
    return pid2images, final_desc


if __name__ == "__main__":
    ds = RobotCarDataset(ds_dir="datasets/robotcar")
    reduce_visible_set(ds)
