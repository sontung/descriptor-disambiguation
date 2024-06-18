import argparse
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from tqdm import tqdm

import dd_utils
from ace_util import read_and_preprocess
from dataset import CMUDataset, RobotCarDataset

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def run_d2_detector_on_all_images(ds_dir, out_dir):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "d2net"

    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    for slice_ in TEST_SLICES:
        print(f"Processing slice {slice_}")
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)
        for ds_ in [train_ds_, test_ds_]:
            out_dir_path = Path(f"{out_dir}/{ds_.ds_type}")
            out_dir_path.mkdir(parents=True, exist_ok=True)

            if ds_.train:
                features_path = (
                    f"{out_dir}/{ds_.ds_type}/{local_desc_model}_features_train.h5"
                )
            else:
                features_path = (
                    f"{out_dir}/{ds_.ds_type}/{local_desc_model}_features_test.h5"
                )
            if Path(features_path).is_file():
                print(f"Skipping {features_path}")
                continue

            print(f"Processing {features_path}")
            features_h5 = h5py.File(str(features_path), "a", libver="latest")

            with torch.no_grad():
                for example in tqdm(ds_, desc="Detecting features"):
                    if example is None:
                        continue
                    name = example[1]
                    image, scale = read_and_preprocess(name, conf_ns)
                    pred = encoder(
                        {"image": torch.from_numpy(image).unsqueeze(0).cuda()}
                    )
                    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                    dict_ = {
                        "scale": scale,
                        "keypoints": pred["keypoints"],
                        "descriptors": pred["descriptors"],
                    }
                    dd_utils.write_to_h5_file(features_h5, name, dict_)
            features_h5.close()


def run_d2_detector_on_all_images_robot_car(ds_dir, out_dir):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "d2net"

    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    train_ds_ = RobotCarDataset(ds_dir=ds_dir)
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=True)
    for ds_ in [train_ds_, test_ds_]:
        out_dir_path = Path(f"{out_dir}/{ds_.ds_type}")
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if ds_.train:
            features_path = (
                f"{out_dir}/{ds_.ds_type}/{local_desc_model}_features_train.h5"
            )
        else:
            features_path = (
                f"{out_dir}/{ds_.ds_type}/{local_desc_model}_features_test.h5"
            )
        if Path(features_path).is_file():
            print(f"Skipping {features_path}")
            continue

        print(f"Processing {features_path}")
        features_h5 = h5py.File(str(features_path), "a", libver="latest")

        with torch.no_grad():
            for example in tqdm(ds_, desc="Detecting features"):
                if example is None:
                    continue
                name = example[1]
                image, scale = read_and_preprocess(name, conf_ns)
                pred = encoder({"image": torch.from_numpy(image).unsqueeze(0).cuda()})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
                dict_ = {
                    "scale": scale,
                    "keypoints": pred["keypoints"],
                    "descriptors": pred["descriptors"],
                }
                dd_utils.write_to_h5_file(features_h5, name, dict_)
        features_h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/cmu_extended",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/n11373598/hpc-home/work/descriptor-disambiguation/output",
        help="Path to the dataset, default: %(default)s",
    )
    args = parser.parse_args()

    run_d2_detector_on_all_images(args.dataset, args.out)
    # run_d2_detector_on_all_images_robot_car(args.dataset, args.out)
