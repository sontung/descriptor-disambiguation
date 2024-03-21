from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

import dd_utils
from dataset import AachenDataset, CMUDataset
from trainer import (
    BaseTrainer,
    ConcatenateTrainer,
    GlobalDescriptorOnlyTrainer,
    MixVPROnlyTrainer,
    MeanOfLocalDescriptorsTrainer, CMUTrainer,
)


TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def use_r2d2(train_ds_, test_ds_, using_global_descriptors):
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
    trainer_.clear()
    del trainer_

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


def use_d2(train_ds_, test_ds_, using_global_descriptors):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "d2net-ss"
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
    trainer_ = BaseTrainer(
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
    trainer_.evaluate()
    del trainer_


def use_superpoint(train_ds_, test_ds_, using_global_descriptors):
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
    trainer_ = BaseTrainer(
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
    trainer_.evaluate()
    del trainer_


if __name__ == "__main__":
    train_ds = CMUDataset(ds_dir="datasets/datasets/cmu_extended/slice2")
    test_ds = CMUDataset(ds_dir="datasets/datasets/cmu_extended/slice2", train=False)

    use_r2d2(train_ds, test_ds, False)

