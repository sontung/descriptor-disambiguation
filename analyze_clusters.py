import pickle

import h5py
import numpy as np
from pykdtree.kdtree import KDTree

import dd_utils
from tqdm import tqdm

from dataset import RobotCarDataset


def perform_clustering(name, nb_clusters=10000):
    arr = np.load(name)
    ids, _ = dd_utils.cluster_by_faiss_kmeans(arr, nb_clusters, verbose=True)
    return ids


def main():

    afile = open("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2ind_debug.pkl", "rb")
    pid2ind = pickle.load(afile)
    afile.close()
    nb_clusters=10
    desc_global = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_desc_debug.npy"
    desc_local = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/pid2mean_desc_debug_False_True.npy"
    local_ids = perform_clustering(desc_local)
    global_ids = perform_clustering(desc_global)

    features_h5 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/d2net_features_test.h5", "r")
    global_features_h5 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/salad_8448_8448_desc_test.h5", "r")

    pgt_matches = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/matches2d_3d.h5", "r")
    ind2pid = {ind: pid for pid, ind in pid2ind.items()}
    mean_acc = []
    test_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=False, evaluate=True)

    for example in tqdm(test_ds_, desc="Computing pose for test set"):
        name = example[1]
        image_name_wo_dir = name.split(test_ds_.images_dir_str)[-1][1:]
        keypoints, descriptors = self.process_descriptor(
            name, features_h5, global_features_h5, gpu_index_flat_for_image_desc
        )

        uv_arr, xyz_pred, indices = self.legal_predict(
            keypoints,
            descriptors,
            gpu_index_flat,
        )

        data = pgt_matches[image_name_wo_dir]
        uv_arr_pgt = np.array(data["uv"])
        pid_list_pgt = np.array(data["pid"])
        tree = KDTree(uv_arr_pgt)
        dis, ind_sub1 = tree.query(uv_arr, 1)
        mask = dis < 1
        pid_list_pred = np.array([ind2pid[ind] for ind in indices[mask]])
        diff = pid_list_pred - pid_list_pgt[ind_sub1[mask]]
        acc = np.sum(diff == 0) / diff.shape[0]
        mean_acc.append(acc)
    features_h5.close()
    global_features_h5.close()

    return


if __name__ == "__main__":
    main()
