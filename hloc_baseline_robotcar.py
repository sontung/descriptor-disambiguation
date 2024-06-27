import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import poselib
import torch
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
)
from hloc.pipelines.RobotCar import colmap_from_nvm
from tqdm import tqdm
import os
import dd_utils
from dataset import RobotCarDataset
from trainer import retrieve_pid


def run_retrieval(img_dir, out_dir, num_loc, db_images):
    images = glob.glob(f"{img_dir}/*/*/*", recursive=True)

    feature_path = f"{out_dir}/global_feats_salad.h5"
    if not os.path.isfile(feature_path):
        from salad_model import SaladModel

        encoder_global = SaladModel()
        with h5py.File(feature_path, "a", libver="latest") as fd:
            for img in tqdm(images, desc="Running global desc"):
                with torch.no_grad():
                    image_descriptor = encoder_global.process(img)
                name = img.split(str(img_dir))[-1][1:]
                if name in fd:
                    del fd[name]
                dict_ = {"global_descriptor": image_descriptor}
                grp = fd.create_group(name)
                for k, v in dict_.items():
                    grp.create_dataset(k, data=v)
    loc_pairs = f"{out_dir}/pairs-query-salad-{num_loc}.txt"
    if not os.path.isfile(loc_pairs):
        pairs_from_retrieval.main(
            feature_path,
            loc_pairs,
            num_loc,
            db_model=db_images,
        )
    return Path(loc_pairs)


def process_db_global_desc(img_dir, out_dir):
    images = glob.glob(f"{img_dir}/*/*/*", recursive=True)

    feature_path = f"{out_dir}/global_feats_salad.h5"
    if not os.path.isfile(feature_path):
        from salad_model import SaladModel

        encoder_global = SaladModel()
        with h5py.File(feature_path, "a", libver="latest") as fd:
            for img in tqdm(images, desc="Running global desc"):
                with torch.no_grad():
                    image_descriptor = encoder_global.process(img)
                name = img.split(str(img_dir))[-1][1:]
                if name in fd:
                    del fd[name]
                dict_ = {"global_descriptor": image_descriptor}
                grp = fd.create_group(name)
                for k, v in dict_.items():
                    grp.create_dataset(k, data=v)
    return feature_path


def perform_retrieval(feature_path, out_dir, num_loc, db_images):
    loc_pairs = f"{out_dir}/pairs-query-salad-{num_loc}.txt"
    pairs_from_retrieval.main(
        feature_path,
        loc_pairs,
        num_loc,
        db_model=db_images,
    )
    return Path(loc_pairs)


def perform_feature_matching(matcher_conf, loc_pairs, feature_conf, outputs):
    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf, outputs)
    return loc_matches


def db_feature_detection(feature_conf, images, outputs):
    features = extract_features.main(feature_conf, images, outputs)
    return features


def compute_pose(train_ds_, test_ds_, features_h5, matches_h5, result_file):
    failed = 0
    for example in tqdm(test_ds_, desc="Computing pose"):
        image_name = example[1]
        image_name_wo_dir = image_name.split(train_ds_.images_dir_str)[-1][1:]
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

    print(f"Failed to localize {failed} images.")


def main_sub(
    train_ds_,
    test_ds_,
    images,
    feature_conf,
    matcher_conf,
    outputs,
    global_features,
    num_loc,
    sift_sfm,
):
    features = db_feature_detection(feature_conf, images, outputs)

    # perform retrieval
    loc_pairs = perform_retrieval(global_features, outputs, num_loc, sift_sfm)

    # match query images with retrieved db images
    loc_matches = perform_feature_matching(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    matches_h5 = h5py.File(
        loc_matches,
        "a",
        libver="latest",
    )
    features_h5 = h5py.File(
        features,
        "a",
        libver="latest",
    )
    result_file = open(
        f"{str(outputs)}/Aachen_v1_1_eval_{str(loc_matches).split('/')[-1].split('.')[0]}.txt",
        "w",
    )

    compute_pose(train_ds_, test_ds_, features_h5, matches_h5, result_file)

    matches_h5.close()
    features_h5.close()
    result_file.close()


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images/"

    outputs = args.outputs  # where everything will be saved
    outputs.mkdir(exist_ok=True, parents=True)
    sift_sfm = outputs / "sfm_sift"

    # pick one of the configurations for extraction and matching
    feature_conf = extract_features.confs["d2net-ss"]
    matcher_conf = match_features.confs["NN-mutual"]

    feature_conf["output"] = feature_conf["model"]["name"]

    colmap_from_nvm.main(
        dataset / "3D-models/all-merged/all.nvm",
        dataset / "3D-models/overcast-reference.db",
        sift_sfm,
    )

    train_ds_ = RobotCarDataset(ds_dir=str(dataset))
    test_ds_ = RobotCarDataset(ds_dir=str(dataset), train=False, evaluate=True)
    global_features = process_db_global_desc(images, outputs)

    main_sub(
        train_ds_,
        test_ds_,
        images,
        feature_conf,
        matcher_conf,
        outputs,
        global_features,
        args.num_loc,
        sift_sfm,
    )


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
        default=20,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
