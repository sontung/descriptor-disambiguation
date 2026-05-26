import os
import glob
from convert_file_size import get_size
from pathlib import Path
from tqdm import tqdm


def get_method_mem(mem_dict, divide=1):
    methods = {
        "hloc": ["uv2xyz", "db_global_desc", "db_images"],
        "light": ["codebook"],
        "heavy": ["codebook", "db_global_desc"],
    }
    method2total_size = {}

    for method in methods:
        mem = 0
        for info in methods[method]:
            mem += mem_dict[info]
        method2total_size[method] = mem
        print(method, get_size(mem / divide))


def aachen():
    out_dir = "/home/n11373598/hpc-home/work/descriptor-disambiguation/output/aachen"
    files_ = {
        "uv2xyz": 4.8e8,
        "db_global_desc": [
            f"{out_dir}/image_desc_name_salad_8448.npy",
            f"{out_dir}/image_desc_salad_8448.npy",
        ],
        "db_images": [
            "/home/n11373598/work/descriptor-disambiguation/datasets/aachen_v1.1/images_upright/db",
            "/home/n11373598/hpc-home/work/descriptor-disambiguation/datasets/aachen_v1.1/images_upright/sequences",
        ],
        "codebook": [
            f"{out_dir}/codebook-d2net-salad_8448.npy",
        ],
    }

    mem_dict = {}
    for file_ in files_:
        values_ = files_[file_]
        if file_ == "db_images":
            mem = 0
            for v in values_:
                rdd = Path(v)
                mem += sum(f.stat().st_size for f in rdd.glob("**/*") if f.is_file())
            mem_dict[file_] = mem
        elif type(values_) is list:
            mem = 0
            for v in values_:
                mem0 = os.path.getsize(v)
                mem += mem0
            mem_dict[file_] = mem
        elif type(values_) is float:
            mem_dict[file_] = values_

    for info in mem_dict:
        print(info, get_size(mem_dict[info]))

    get_method_mem(mem_dict)


def cmu():
    slices = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    files_ = {
        "uv2xyz": [
            f"/work/qvpr/data/raw/2020VisualLocalization/Extended-CMU-Seasons/slice*/sparse/images.bin"
        ],
        "db_global_desc": [
            f"../output/cmu/slice*/image_desc_salad_8448_8448.npy",
            f"../output/cmu/slice*/image_desc_name_salad_8448_8448.npy",
        ],
        "db_images": [
            "/work/qvpr/data/raw/2020VisualLocalization/Extended-CMU-Seasons/slice*/database"
        ],
        "codebook": [
            f"../output/cmu/slice*/codebook-d2net-salad_8448.npy",
        ],
    }

    mem_dict = {}
    for file_ in files_:
        values_ = files_[file_]
        if file_ != "db_images":
            mem = 0
            for v in values_:
                # assert len(glob.glob(v)) == 14, len(glob.glob(v))
                for slice in slices:
                    v2 = v.replace("*", str(slice))
                    mem += os.path.getsize(v2)
            mem_dict[file_] = mem
        else:
            mem = 0
            for v in values_:
                for slice in tqdm(slices):
                    path_to_images = v.replace("*", str(slice))
                    all_images = glob.glob(f"{path_to_images}/*")
                    for img in all_images:
                        mem += os.path.getsize(img)
            mem_dict[file_] = mem

    for info in mem_dict:
        print(info, get_size(mem_dict[info]))
    get_method_mem(mem_dict)
    return


def cambridge():
    global_model = "mixvpr_512"
    local_model = "d2net"
    sequences = [
        "GreatCourt",
        "KingsCollege",
        "OldHospital",
        "ShopFacade",
        "StMarysChurch",
    ]
    files_ = {
        "uv2xyz": [f"../datasets/cambridge/{{ds_name}}/reconstruction.nvm"],
        "db_global_desc": [
            f"../output/{{ds_name}}/image_desc_{global_model}.npy",
            f"../output/{{ds_name}}/image_desc_name_{global_model}.npy",
        ],
        "db_images": ["../datasets/cambridge/{ds_name}/seq*/*.png"],
        "codebook": [
            f"../output/{{ds_name}}/codebook-{local_model}-{global_model}.npy"
        ],
    }

    mem_dict = {}
    for file_ in files_:
        values_ = files_[file_]
        mem = 0
        for seq in sequences:
            for v in values_:
                v = v.format(ds_name=seq)
                if "*" in v:
                    mem += sum([os.path.getsize(du) for du in glob.glob(v)])
                else:
                    mem += os.path.getsize(v)
        mem_dict[file_] = mem
    for info in mem_dict:
        print(info, get_size(mem_dict[info]))
    get_method_mem(mem_dict, 5)

    print()


def robotcar():
    files_ = {
        "uv2xyz": [
            f"/work/qvpr/data/raw/2020VisualLocalization/RobotCar-Seasons/3D-models/all-merged/all.nvm"
        ],
        "db_global_desc": [
            f"../output/robotcar/image_desc_salad_8448.npy",
            f"../output/robotcar/image_desc_name_salad_8448.npy",
        ],
        "db_images": [
            "/work/qvpr/data/raw/2020VisualLocalization/RobotCar-Seasons/images/*/*/*.jpg"
        ],
        "codebook": [
            "../output/robotcar/codebook-d2net-salad_8448.npy",
        ],
    }

    mem_dict = {}
    for file_ in files_:
        values_ = files_[file_]
        mem = 0
        for v in values_:
            # assert len(glob.glob(v)) >= 5, glob.glob(v)
            mem += sum([os.path.getsize(du) for du in glob.glob(v)])
        mem_dict[file_] = mem
        print(file_, mem)
    for info in mem_dict:
        print(info, get_size(mem_dict[info]))
    get_method_mem(mem_dict, 1)

    print()


if __name__ == "__main__":
    robotcar()
    # aachen()
    # cmu()
    # cambridge()
