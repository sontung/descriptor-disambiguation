import argparse
from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from pathlib import Path

import dd_utils
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer


def use_r2d2(root_dir_, using_global_descriptors):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "r2d2"
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    retrieval_model = "eigenplaces"
    model_dict = conf[retrieval_model]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    model_dict.update(
        {"variant": "EigenPlaces", "backbone": "ResNet101", "fc_output_dim": 2048}
    )
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]

    folders = [
        item.name
        for item in Path(root_dir_).iterdir()
        if item.is_dir() and "Cambridge" in item.name
    ]

    results = {}
    for ds_name in folders:
        print(f"Processing {ds_name}")
        train_ds_ = CambridgeLandmarksDataset(
            train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )
        test_ds_ = CambridgeLandmarksDataset(
            train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )

        trainer_ = CambridgeLandmarksTrainer(
            train_ds_,
            test_ds_,
            128,
            2048,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )
        t_err, r_err = trainer_.evaluate()
        print(f"    median translation error = {t_err}")
        print(f"    median rotation error = {r_err}")
        results[ds_name] = (t_err, r_err)
        del trainer_
    print(f"Results: (translation/rotation) {local_desc_model} {retrieval_model}")
    t0, r0 = 0, 0
    for ds in results:
        t_err, r_err = results[ds]
        t0 += t_err
        r0 += r_err
        print(f"    {ds} {t_err:.3f}/{r_err:.3f}")
    print(f"    Avg. {t0/len(results):.3f}/{r0/len(results):.3f}")


def use_d2(root_dir_, using_global_descriptors):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "d2net"
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    retrieval_model = "eigenplaces"
    model_dict = conf[retrieval_model]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    model_dict.update(
        {"variant": "EigenPlaces", "backbone": "ResNet101", "fc_output_dim": 512}
    )
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]
    folders = [
        item.name
        for item in Path(root_dir_).iterdir()
        if item.is_dir() and "Cambridge" in item.name
    ]

    results = {}
    for ds_name in folders:
        print(f"Processing {ds_name}")
        train_ds_ = CambridgeLandmarksDataset(
            train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )
        test_ds_ = CambridgeLandmarksDataset(
            train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )

        trainer_ = CambridgeLandmarksTrainer(
            train_ds_,
            test_ds_,
            512,
            512,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )
        t_err, r_err = trainer_.evaluate()
        print(f"    median translation error = {t_err}")
        print(f"    median rotation error = {r_err}")
        results[ds_name] = (t_err, r_err)
        del trainer_
    print(f"Results: (translation/rotation) {local_desc_model} {retrieval_model}")
    t0, r0 = 0, 0
    for ds in results:
        t_err, r_err = results[ds]
        t0 += t_err
        r0 += r_err
        print(f"    {ds} {t_err:.3f}/{r_err:.3f}")
    print(f"    Avg. {t0/len(results):.3f}/{r0/len(results):.3f}")


def use_superpoint(root_dir_, using_global_descriptors):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "superpoint"
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    retrieval_model = "eigenplaces"
    model_dict = conf[retrieval_model]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    model_dict.update(
        {"variant": "EigenPlaces", "backbone": "ResNet101", "fc_output_dim": 2048}
    )
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]
    folders = [
        item.name
        for item in Path(root_dir_).iterdir()
        if item.is_dir() and "Cambridge" in item.name
    ]

    results = {}
    for ds_name in folders:
        print(f"Processing {ds_name}")
        train_ds_ = CambridgeLandmarksDataset(
            train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )
        test_ds_ = CambridgeLandmarksDataset(
            train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )

        trainer_ = CambridgeLandmarksTrainer(
            train_ds_,
            test_ds_,
            256,
            2048,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )
        t_err, r_err = trainer_.evaluate()
        print(f"    median translation error = {t_err}")
        print(f"    median rotation error = {r_err}")
        results[ds_name] = (t_err, r_err)
        del trainer_
    print(f"Results: (translation/rotation) {local_desc_model} {retrieval_model}")
    t0, r0 = 0, 0
    for ds in results:
        t_err, r_err = results[ds]
        t0 += t_err
        r0 += r_err
        print(f"    {ds} {t_err:.3f}/{r_err:.3f}")
    print(f"    Avg. {t0/len(results):.3f}/{r0/len(results):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../ace/datasets",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument("--use_global", type=int, default=1)
    parser.add_argument(
        "--local_desc",
        type=str,
        default="r2d2",
    )
    args = parser.parse_args()
    if args.local_desc == "r2d2":
        use_r2d2(args.dataset, bool(args.use_global))
    elif args.local_desc == "superpoint":
        use_superpoint(args.dataset, bool(args.use_global))
    elif args.local_desc == "d2":
        use_d2(args.dataset, bool(args.use_global))
