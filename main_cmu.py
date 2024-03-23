import argparse
from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

import dd_utils
from dataset import CMUDataset
from trainer import (
    CMUTrainer,
)

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def use_r2d2(ds_dir, using_global_descriptors):
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

    results = []
    for slice_ in TEST_SLICES:
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)

        trainer_ = CMUTrainer(
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
        query_results = trainer_.evaluate()
        results.extend(query_results)
        trainer_.clear()
        del trainer_
        train_ds_.clear()

    if using_global_descriptors:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}_{retrieval_model}.txt",
            "w",
        )
    else:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}.txt",
            "w",
        )
    for line in results:
        print(line, file=result_file)
    result_file.close()


def use_d2(ds_dir, using_global_descriptors):
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
        {"variant": "EigenPlaces", "backbone": "ResNet101", "fc_output_dim": 2048}
    )
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]

    results = []
    for slice_ in TEST_SLICES:
        print(f"Processing slice {slice_}")
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)

        trainer_ = CMUTrainer(
            train_ds_,
            test_ds_,
            512,
            2048,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )
        query_results = trainer_.evaluate()
        results.extend(query_results)
        trainer_.clear()
        del trainer_
        train_ds_.clear()

    if using_global_descriptors:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}_{retrieval_model}.txt",
            "w",
        )
    else:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}.txt",
            "w",
        )
    for line in results:
        print(line, file=result_file)
    result_file.close()


def use_superpoint(ds_dir, using_global_descriptors):
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
    results = []

    for slice_ in TEST_SLICES:
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)

        trainer_ = CMUTrainer(
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
        query_results = trainer_.evaluate()
        results.extend(query_results)
        trainer_.clear()
        del trainer_
        train_ds_.clear()

    if using_global_descriptors:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}_{retrieval_model}.txt",
            "w",
        )
    else:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}.txt",
            "w",
        )
    for line in results:
        print(line, file=result_file)
    result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/datasets/cmu_extended",
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
