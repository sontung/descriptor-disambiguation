import h5py
import numpy as np
from pykdtree.kdtree import KDTree
from dataset import RobotCarDataset

features1 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/output/robotcar/d2net_features_test.h5", "r")
pgt_matches = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/matches2d_3d.h5", "r")
features2 = h5py.File("/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar/d2net.h5", "r")
test_ds_ = RobotCarDataset(ds_dir="datasets/robotcar", train=False, evaluate=True)

for example in test_ds_:
    name = example[1]
    name2 = "/".join(name.split("/")[-2:])
    image_name_wo_dir = name.split(test_ds_.images_dir_str)[-1][1:]
    kp2 = np.array(features2[image_name_wo_dir]["keypoints"])
    kp1 = np.array(features1[name2]["keypoints"])
    kp3 = np.array(pgt_matches[image_name_wo_dir]["uv"])

    tree = KDTree(kp3)
    dis1, _ = tree.query(kp1)
    dis2, _ = tree.query(kp2)
    print(np.sum(dis1<=0.1))
    dis3 = KDTree(kp2).query(kp1, 1)[0]
    break

features2.close()
features1.close()
pgt_matches.close()