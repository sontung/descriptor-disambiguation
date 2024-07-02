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

    afile = open("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2ind_debug.pkl", "rb")
    pid2ind = pickle.load(afile)
    afile.close()
    desc_global = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_desc_debug.npy"
    desc_local = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_desc_debug_False_True.npy"

    features_h5 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/d2net_features_test.h5", "r")
    global_features_h5 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/salad_8448_8448_desc_test.h5", "r")

    pgt_matches = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/matches2d_3d.h5", "r")
    pred_matches = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/matches2d3dTrue_True.npy", "r")
    ind2pid = {ind: pid for pid, ind in pid2ind.items()}
    test_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=False, evaluate=True)
    train_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=True, evaluate=False)

    # local_indices, local_cluster_machine = perform_clustering(desc_local, nb_clusters=10)
    # desc_local_arr = np.load(desc_local) 
    # desc_local_arr = np.load(desc_global)

    mean_acc = []
    all_dis = []
    count = 0
    for example in tqdm(test_ds_, desc="Computing pose for test set"):
        name = example[1]
        image_name_wo_dir = name.split(test_ds_.images_dir_str)[-1][1:]

        img_id = "/".join(name.split("/")[-2:])
        local_desc_curr = np.array(features_h5[img_id]["descriptors"]).T
        global_desc_curr = np.array(global_features_h5[img_id]["global_descriptor"]).reshape(-1, 1)
        final_desc_curr = 0.3 * local_desc_curr + (1 - 0.3) * global_desc_curr[:local_desc_curr.shape[0]]

        data = pgt_matches[image_name_wo_dir]
        data2 = pred_matches[image_name_wo_dir]
        uv_arr_pgt = np.array(data["uv"])
        uv_arr = np.array(data2["uv"])
        pid_list_pgt = np.array(data["pid"])
        ind_list = np.array(data2["pid"])
        mask1 = [True if pid in pid2ind else False for pid in pid_list_pgt]
        pid_list_pgt = pid_list_pgt[mask1]
        uv_arr_pgt = uv_arr_pgt[mask1]
        tree = KDTree(uv_arr_pgt)
        dis, ind_sub1 = tree.query(uv_arr, 1)
        mask = dis < 1
        pid_list_pred = np.array([ind2pid[ind] for ind in ind_list[mask]])
        diff = pid_list_pred - pid_list_pgt[ind_sub1[mask]]
        acc = np.sum(diff == 0) / diff.shape[0]

        mask2 = diff != 0
        if np.sum(mask2) == 0:
            print(acc)
            continue
        indices_pgt_2 = [pid2ind[pid] for pid in pid_list_pgt[ind_sub1[mask]][mask2]]
        indices_pred_2 = [pid2ind[pid] for pid in pid_list_pred[mask2]]
        dis = np.mean(
            np.abs(train_ds_.xyz_arr[pid_list_pred[mask2]]-
                   train_ds_.xyz_arr[pid_list_pgt[ind_sub1[mask]][mask2]]),
            1)
        all_dis.extend(dis)

        img = cv2.imread(name)
        uvs = uv_arr[mask][mask2].astype(int)
        # for u, v in uvs[dis<1]:
        #     cv2.circle(img, (u, v), 5, (255, 0, 0), -1)
        # for u, v in uvs[dis>5]:
        #     cv2.circle(img, (u, v), 5, (0, 0, 255), -1)

        for u, v in uv_arr_pgt.astype(int):
            cv2.circle(img, (u, v), 5, (255, 0, 0), -1)
        for u, v in uv_arr.astype(int):
            cv2.circle(img, (u, v), 5, (0, 255, 0), -1)
        cv2.imwrite(f"debug/img{count}.png", img)
        count += 1

    print(np.mean(mean_acc))
    features_h5.close()
    global_features_h5.close()
    pgt_matches.close()
    pred_matches.close()

    return


if __name__ == "__main__":
    main()
