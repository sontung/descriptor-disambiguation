import numpy as np
import sys
import faiss
import hurry.filesize
import pickle


all_desc = np.load("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/aachen/codebook.npy")
afile = open("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/aachen/all_pids.pkl", "rb")
all_pid = pickle.load(afile)
afile.close()

index = faiss.IndexFlatL2(all_desc.shape[1])  # build the index
res = faiss.StandardGpuResources()
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index_flat.add(all_desc)
print(hurry.filesize.size(sys.getsizeof(gpu_index_flat)))
print(hurry.filesize.size(sys.getsizeof(all_desc)))
print(hurry.filesize.size(sys.getsizeof(all_desc.astype(np.float32))))

print()