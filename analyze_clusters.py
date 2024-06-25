import numpy as np
import dd_utils


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

    for i in range(nb_clusters):
        mask = local_ids==i
        global_cid = global_ids[mask]
        break

    return


if __name__ == "__main__":
    main()
