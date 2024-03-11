from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

import dd_utils
from dataset import AachenDataset
from trainer import BaseTrainer


def use_r2d2(using_global_descriptors):
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
    trainer_ = BaseTrainer(
        AachenDataset(),
        AachenDataset(train=False),
        128,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        using_global_descriptors,
    )
    trainer_.evaluate()


def use_d2(using_global_descriptors):
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
        AachenDataset(),
        AachenDataset(train=False),
        512,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        using_global_descriptors,
    )
    trainer_.evaluate()


if __name__ == "__main__":
    use_r2d2(True)
    use_d2(True)
