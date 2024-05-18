import os
import glob
import hurry.filesize
from pathlib import Path

out_dir = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/aachen"
files_ = {
    "uv2xyz": [f"/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/aachen_v1.1/d2net_nn.h5"],
    "db_global_desc": [f"{out_dir}/image_desc_eigenplaces2048_2048.npy",
                       f"{out_dir}/image_desc_name_eigenplaces2048_2048.npy"],
    "db_images": ["/home/n11373598/work/descriptor-disambiguation/datasets/aachen_v1.1/images_upright/db",
                  "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/aachen_v1.1/images_upright/sequences"],
    "codebook": [f"{out_dir}/codebook-d2net-eigenplaces_ResNet101_2048.npy",
                 f"{out_dir}/pid2ind-d2net-eigenplaces_ResNet101_2048.pkl"]
}


mem_dict = {}
for file_ in files_:
    values_ = files_[file_]
    if file_ == "db_images":
        mem = 0
        for v in values_:
            rdd = Path(v)
            mem += sum(f.stat().st_size for f in rdd.glob('**/*') if f.is_file())
        mem_dict[file_] = mem
    elif type(values_) is list:
        mem = 0
        for v in values_:
            mem0 = os.path.getsize(v)
            mem += mem0
        mem_dict[file_] = mem

for info in mem_dict:
    print(info, hurry.filesize.size(mem_dict[info]))

methods = {
    "hloc": ["uv2xyz", "db_global_desc", "db_images"],
    "light": ["codebook"],
    "heavy": ["codebook", "db_global_desc"]
}
method2total_size = {}

for method in methods:
    mem = 0
    for info in methods[method]:
        mem += mem_dict[info]
    method2total_size[method] = mem
    print(method, hurry.filesize.size(mem))

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
width = 0.6
file2values = {
    "uv2xyz": np.array([1, 0, 0]),
    "db_global_desc": np.array([1, 0, 1]),
    "codebook": np.array([0, 1, 1]),
    "db_images": np.array([1, 0, 0]),
}

bottom = np.zeros(3)

for file_, file_size in file2values.items():
    p = ax.bar(list(methods.keys()), file_size*mem_dict[file_], width, label=file_, bottom=bottom)
    bottom += file_size*mem_dict[file_]

    ax.bar_label(p, label_type='center')
ax.legend()

plt.savefig("mem_fig.png")

print()