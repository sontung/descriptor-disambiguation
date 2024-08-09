import h5py
import numpy as np
import faiss


def roll_matrix(all_desc, nd, norm=False):
    nb = all_desc.shape[0]
    if nd == all_desc.shape[1]:
        return all_desc
    all_desc_rolled = np.zeros((nb, nd))
    count = 0
    for i in range(0, all_desc.shape[1], nd):
        start = i
        end = min(i + nd, all_desc.shape[1])
        all_desc_rolled[:, : end - start] += all_desc[:, start:end]
        count += 1
    if norm:
        all_desc_rolled /= count
    return all_desc_rolled


desc0 = np.load(
    "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/image_desc_salad_8448.npy"
)
print(desc0.shape)
desc1_file = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/salad_8448_8448_desc_test.h5"
# test_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=False, evaluate=True)

with h5py.File(desc1_file, "r") as fd:
    desc1 = np.zeros((len(fd["rear"]), desc0.shape[1]))
    for idx, k in enumerate(fd["rear"].keys()):
        v = np.array(fd[f"rear/{k}"]["global_descriptor"])
        desc1[idx] = v

# desc0 = desc0.astype(np.float32)
# desc1 = desc1.astype(np.float32)

index2 = faiss.IndexFlat(desc0.shape[1], faiss.METRIC_L2)  # build the index
res2 = faiss.StandardGpuResources()
gpu_index_flat_for_image_desc = faiss.index_cpu_to_gpu(res2, 0, index2)
gpu_index_flat_for_image_desc.add(desc0)
dis, ind1 = gpu_index_flat_for_image_desc.search(desc1, 1)


import umap

transformer = umap.UMAP(n_components=512, n_neighbors=50)
desc0_red = transformer.fit_transform(desc0)
desc1_red = transformer.transform(desc1)

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

transformer = SparseRandomProjection(n_components=512)
desc0_red = transformer.fit_transform(desc0)
desc1_red = transformer.transform(desc1)

# desc0r = roll_matrix(desc0, 512, norm=True).astype(np.float32)
# desc1r = roll_matrix(desc1, 512, norm=True).astype(np.float32)
index2 = faiss.IndexFlat(512, faiss.METRIC_L2)
gpu_index_flat_for_image_desc = faiss.index_cpu_to_gpu(res2, 0, index2)
gpu_index_flat_for_image_desc.add(desc0_red)
dis, ind2 = gpu_index_flat_for_image_desc.search(desc1_red, 1)
diff = ind2 - ind1
print(np.sum(diff == 0) / diff.shape[0])

var_val = np.var(desc0, 0)
indices = np.argsort(var_val)[-512:]

for _ in range(10):
    indices = np.arange(desc0.shape[1])
    np.random.shuffle(indices)
    indices = indices[:512]
    index2 = faiss.IndexFlat(len(indices), faiss.METRIC_L2)
    gpu_index_flat_for_image_desc = faiss.index_cpu_to_gpu(res2, 0, index2)
    gpu_index_flat_for_image_desc.add(desc0[:, indices])
    dis, ind2 = gpu_index_flat_for_image_desc.search(desc1[:, indices], 1)
    diff = ind2 - ind1
    print(np.sum(diff == 0) / diff.shape[0])
