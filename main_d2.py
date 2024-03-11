import logging
from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load

import dd_utils
from dataset import AachenDataset
from trainer import BaseTrainer


class TrainerACE(BaseTrainer):
    def __init__(self, using_global_descriptors=True):
        conf, default_conf = dd_utils.hloc_conf_for_all_models()
        self.local_desc_model = "d2net-ss"
        model_dict = conf[self.local_desc_model]["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        encoder = Model(model_dict).eval().to(device)
        conf_ns = SimpleNamespace(**{**default_conf, **conf})
        conf_ns.grayscale = conf[self.local_desc_model]["preprocessing"]["grayscale"]
        conf_ns.resize_max = conf[self.local_desc_model]["preprocessing"]["resize_max"]

        self.retrieval_model = "eigenplaces"
        model_dict = conf[self.retrieval_model]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        model_dict.update(
            {"variant": "EigenPlaces", "backbone": "ResNet101", "fc_output_dim": 2048}
        )
        encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf[self.retrieval_model]["preprocessing"][
            "resize_max"
        ]
        super().__init__(
            AachenDataset(),
            AachenDataset(train=False),
            512,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = TrainerACE(using_global_descriptors=False)
    trainer.evaluate()
