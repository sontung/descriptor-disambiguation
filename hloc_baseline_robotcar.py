import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import poselib
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
)
from hloc.pipelines.RobotCar import colmap_from_nvm
from tqdm import tqdm

from dataset import RobotCarDataset
from trainer import retrieve_pid


def generate_query_list(dataset, image_dir, path):
    h, w = 1024, 1024
    intrinsics_filename = "intrinsics/{}_intrinsics.txt"
    cameras = {}
    for side in ["left", "right", "rear"]:
        with open(dataset / intrinsics_filename.format(side), "r") as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ["SIMPLE_RADIAL", w, h, fx, cx, cy, 0.0]
            cameras[side] = [str(p) for p in params]

    queries = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    queries = [
        Path(q).relative_to(image_dir.parents[0]).as_posix() for q in sorted(queries)
    ]

    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images/"

    outputs = args.outputs  # where everything will be saved
    outputs.mkdir(exist_ok=True, parents=True)
    sift_sfm = outputs / "sfm_sift"

    loc_pairs = outputs / f"pairs-query{args.num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["eigenplaces"]
    feature_conf = extract_features.confs["d2net-ss"]
    matcher_conf = match_features.confs["NN-mutual"]

    feature_conf["output"] = feature_conf["model"]["name"]

    # colmap_from_nvm.main(
    #     dataset / "3D-models/all-merged/all.nvm",
    #     dataset / "3D-models/overcast-reference.db",
    #     sift_sfm,
    # )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        db_model=sift_sfm,
    )

    features = extract_features.main(feature_conf, images, outputs)

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    train_ds_ = RobotCarDataset(ds_dir=str(dataset))
    test_ds_ = RobotCarDataset(ds_dir=str(dataset), train=False, evaluate=True)

    result_file = open(
        f"{str(outputs)}/Robotcar_eval_{str(loc_matches).split('/')[-1].split('.')[0]}.txt",
        "w",
    )
    matches_h5 = h5py.File(
        loc_matches,
        "r",
        libver="latest",
    )
    features_h5 = h5py.File(
        features,
        "r",
        libver="latest",
    )
    img_dir_str = train_ds_.images_dir_str

    failed = 0
    for example in tqdm(test_ds_, desc="Computing pose"):
        image_name = example[1]
        image_name_wo_dir = image_name.split(img_dir_str)[-1][1:]
        image_name_for_matching_db = image_name_wo_dir.replace("/", "-")
        data = matches_h5[image_name_for_matching_db]
        matches_2d_3d = []
        for db_img in data:
            matches = data[db_img]
            indices = np.array(matches["matches0"])
            mask0 = indices > -1
            if np.sum(mask0) < 10:
                continue
            if len(db_img.split("-")) == 3:
                db_img_normal = db_img.replace("-", "/")
            else:
                db_img_normal = db_img.replace("-", "/").replace("/", "-", 1)
            uv1 = np.array(features_h5[db_img_normal]["keypoints"])
            uv1 = uv1[indices[mask0]]

            db_img_id = train_ds_.name2image[f"./{db_img_normal.replace('jpg', 'png')}"]
            pid_list = train_ds_.image2points[db_img_id]
            uv_gt = np.array(train_ds_.image2uvs[db_img_id])

            selected_pid, mask, ind = retrieve_pid(pid_list, uv_gt, uv1)
            idx_arr, ind2 = np.unique(ind[mask], return_index=True)

            matches_2d_3d.append([mask0, idx_arr, selected_pid[ind2]])

        uv0 = np.array(features_h5[image_name_wo_dir]["keypoints"])
        index_arr_for_kp = np.arange(uv0.shape[0])
        all_matches = [[], [], []]
        for mask0, idx_arr, pid_list in matches_2d_3d:
            uv0_selected = uv0[mask0][idx_arr]
            indices = index_arr_for_kp[mask0][idx_arr]
            all_matches[0].append(uv0_selected)
            all_matches[1].extend(pid_list)
            all_matches[2].extend(indices)

        if len(all_matches[1]) < 10:
            qvec = "0 0 0 1"
            tvec = "0 0 0"
            failed += 1
        else:
            uv_arr = np.vstack(all_matches[0])
            xyz_pred = train_ds_.xyz_arr[all_matches[1]]
            camera = example[6]

            camera_dict = {
                "model": camera.model.name,
                "height": camera.height,
                "width": camera.width,
                "params": camera.params,
            }
            pose, info = poselib.estimate_absolute_pose(
                uv_arr,
                xyz_pred,
                camera_dict,
            )

            qvec = " ".join(map(str, pose.q))
            tvec = " ".join(map(str, pose.t))

        image_id = "/".join(example[2].split("/")[1:])
        print(f"{image_id} {qvec} {tvec}", file=result_file)

    matches_h5.close()
    features_h5.close()
    result_file.close()
    print(f"Failed to localize {failed} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/robotcar",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
