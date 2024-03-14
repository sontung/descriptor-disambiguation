from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

import dd_utils
from dataset import AachenDataset
from trainer import (
    BaseTrainer,
    ConcatenateTrainer,
    GlobalDescriptorOnlyTrainer,
    MixVPROnlyTrainer,
    MeanOfLocalDescriptorsTrainer,
)


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
    trainer_ = BaseTrainer(
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
    trainer_.evaluate()
    del trainer_


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


def use_r2d2_concat(train_ds_, test_ds_):
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
    trainer_ = ConcatenateTrainer(
        train_ds_,
        test_ds_,
        128,
        2048,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
    )
    trainer_.evaluate()
    del trainer_


def use_d2_concat(train_ds_, test_ds_):
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
    trainer_ = ConcatenateTrainer(
        train_ds_,
        test_ds_,
        512,
        2048,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        True
    )
    trainer_.evaluate()
    del trainer_


def use_eigen(train_ds_, test_ds_):
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
    trainer_ = GlobalDescriptorOnlyTrainer(
        train_ds_,
        test_ds_,
        128,
        2048,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
    )
    trainer_.evaluate()
    del trainer_


def use_netvlad(train_ds_, test_ds_):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "r2d2"
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    retrieval_model = "netvlad"
    model_dict = conf[retrieval_model]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]
    trainer_ = GlobalDescriptorOnlyTrainer(
        train_ds_,
        test_ds_,
        128,
        4096,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
    )
    trainer_.evaluate()
    del trainer_


def use_mixvpr(train_ds_, test_ds_):
    trainer_ = MixVPROnlyTrainer(
        train_ds_,
        test_ds_,
        128,
        4096,
        None,
        None,
        None,
        None,
    )
    trainer_.evaluate()
    del trainer_


def use_mean_of_local_descriptors(train_ds_, test_ds_):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    local_desc_model = "r2d2"
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    retrieval_model = "netvlad"
    model_dict = conf[retrieval_model]["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder_global = Model(model_dict).eval().to(device)
    conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
    conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"]["resize_max"]
    trainer_ = MeanOfLocalDescriptorsTrainer(
        train_ds_,
        test_ds_,
        128,
        128,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
    )
    trainer_.evaluate()
    del trainer_


if __name__ == "__main__":
    train_ds = AachenDataset()
    test_ds = AachenDataset(train=False)

    # use_netvlad(train_ds, test_ds)
    # use_mixvpr(train_ds, test_ds)
    # use_mean_of_local_descriptors(train_ds, test_ds)
    use_d2_concat(train_ds, test_ds)

    # use_r2d2(train_ds, test_ds, True)
    # use_d2(train_ds, test_ds, True)
    # use_superpoint(train_ds, test_ds, True)

    # use_r2d2(train_ds, test_ds, False)
    # use_d2(train_ds, test_ds, False)
    # use_superpoint(train_ds, test_ds, False)
