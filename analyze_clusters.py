import numpy as np
import dd_utils
from tqdm import tqdm

def perform_clustering(name, nb_clusters=10000):
    arr = np.load(name)
    ids, _ = dd_utils.cluster_by_faiss_kmeans(arr, nb_clusters, verbose=True)
    return ids


def main():
    nb_clusters=10000
    xyz_global = (
        "/home/n11373598/work/descriptor-disambiguation/output/robotcar/xyz-global.npy"
    )
    desc_global = "/home/n11373598/work/descriptor-disambiguation/output/robotcar/codebook-global.npy"
    xyz_local = (
        "/home/n11373598/work/descriptor-disambiguation/output/robotcar/xyz-local.npy"
    )
    desc_local = "/home/n11373598/work/descriptor-disambiguation/output/robotcar/codebook-local.npy"
    xyz = np.load(xyz_global)
    local_ids = perform_clustering(desc_local)
    global_ids = perform_clustering(desc_global)

    var_list = []
    pid_list = np.arange(xyz.shape[0])
    remove_pids = []
    for i in tqdm(range(nb_clusters)):
        mask = local_ids==i
        if np.sum(mask) <= 2:
            continue
        if np.mean(np.var(xyz[mask], 0)) > 10:
            xyz_curr = xyz[mask]
            pid_curr = pid_list[mask]
            global_cids = np.unique(global_ids[mask])
            for gc in global_cids:
                mask2 = global_ids[mask] == gc
                if np.mean(np.var(xyz_curr[mask2], 0)) > 10:
                    remove_pids.extend(pid_curr[mask2])
    np.save("output/robotcar/bad_pids.npy", np.unique(remove_pids))
    return


if __name__ == "__main__":
    main()
