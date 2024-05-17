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
index.add(all_desc)
faiss.write_index(index, "large.index")

d = 512        # Dimension (length) of vectors.
nlist = 10000  # Number of inverted lists (number of partitions or cells).
nsegment = 16  # Number of segments for product quantization (number of subquantizers).
nbit = 8       # Number of bits to encode each segment.

# Create the index.
coarse_quantizer = faiss.IndexFlatL2(512)
index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, nsegment, nbit)
index.train(all_desc)
index.add(all_desc)
faiss.write_index(index, "large.index")

nlist = 100
m = 8                             # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(all_desc)
index.add(all_desc)

print()