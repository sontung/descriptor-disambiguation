import argparse
from pathlib import Path

import h5py
import numpy as np
import poselib
from hloc import extract_features, match_features
from hloc import pairs_from_retrieval
from tqdm import tqdm

from dataset import AachenDataset
from trainer import retrieve_pid


def perform_retrieval(global_descriptors, loc_pairs, num_loc, sift_sfm):
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        query_prefix="query",
        db_model=sift_sfm,
    )


def perform_feature_matching(matcher_conf, loc_pairs, feature_conf, outputs):
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf, outputs
    )
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
            db_img_normal = db_img.replace("-", "/")
            uv1 = np.array(features_h5[db_img_normal]["keypoints"])
            uv1 = uv1[indices[mask0]]

            db_img_id = train_ds_.image_name2id[db_img_normal]
            pid_list = train_ds_.image_id2pids[db_img_id]
            uv_gt = train_ds_.image_id2uvs[db_img_id]
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
            xyz_pred = np.array([train_ds_.recon_points[pid].xyz for pid in all_matches[1]])
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

        image_id = image_name.split("/")[-1]
        print(f"{image_id} {qvec} {tvec}", file=result_file)
    print(f"Failed to localize {failed} images.")


@profile
def main_sub(train_ds_, test_ds_, feature_conf, retrieval_conf, matcher_conf,
             images, outputs, loc_pairs, num_loc, sift_sfm):
    # extract local features for db images
    features = db_feature_detection(feature_conf, images, outputs)

    # extract global features for db images
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    # perform retrieval
    perform_retrieval(global_descriptors, loc_pairs, num_loc, sift_sfm)

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


@profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen_v1.1",
        # default="/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/aachen_v1.1",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()

    # Setup the paths
    dataset = args.dataset
    images = dataset / "images_upright/"
    sift_sfm = dataset / "3D-models/aachen_v_1_1"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["eigenplaces"]
    feature_conf = extract_features.confs["d2net-ss"]
    matcher_conf = match_features.confs["NN-mutual"]

    # matcher_conf["output"] = matcher_conf['model']['name']
    feature_conf["output"] = feature_conf["model"]["name"]

    outputs = args.outputs  # where everything will be saved
    loc_pairs = (
        outputs / f"pairs-query-{retrieval_conf['model']['name']}-{args.num_loc}.txt"
    )  # top-k retrieved by NetVLAD

    train_ds_ = AachenDataset(ds_dir=dataset)
    test_ds_ = AachenDataset(ds_dir=dataset, train=False)
    main_sub(train_ds_, test_ds_, feature_conf, retrieval_conf, matcher_conf,
             images, outputs, loc_pairs, args.num_loc, sift_sfm)


if __name__ == '__main__':
    main()

