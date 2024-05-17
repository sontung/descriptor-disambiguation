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
    "codebook": [f"{out_dir}/codebook_d2net_eigenplaces_ResNet50_2048_2048.npy",
                 f"{out_dir}/all_pids_d2net_eigenplaces_ResNet50_2048_2048.npy",
                 f"{out_dir}/pid2ind_d2net_eigenplaces_ResNet50_2048_2048.pkl"]
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
for method in methods:
    mem = 0
    for info in methods[method]:
        mem += mem_dict[info]
    print(method, hurry.filesize.size(mem))
