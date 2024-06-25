import numpy as np
import dd_utils


def main():
    xyz_global = (
        "/home/n11373598/work/descriptor-disambiguation/output/robotcar/xyz-global.npy"
    )
    desc_global = "/home/n11373598/work/descriptor-disambiguation/output/robotcar/codebook-global.npy"
    xyz_local = (
        "/home/n11373598/work/descriptor-disambiguation/output/robotcar/xyz-local.npy"
    )
    desc_local = "/home/n11373598/work/descriptor-disambiguation/output/robotcar/codebook-local.npy"
    xyz_global = np.load(xyz_global)
    xyz_local = np.load(xyz_local)
    desc_global = np.load(desc_global)
    desc_local = np.load(desc_local)
    return


if __name__ == "__main__":
    main()
