import argparse

import numpy as np

import dd_utils
from dataset import AachenDataset
from trainer import BaseTrainer


def run_function(
    ds_dir,
    local_desc_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    if using_global_descriptors:
        print(f"Using {local_desc_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_desc_model}")

    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_desc_model, retrieval_model, global_desc_dim
    )
    train_ds_ = AachenDataset(ds_dir=ds_dir)
    test_ds_ = AachenDataset(ds_dir=ds_dir, train=False)
    for lambda_val in np.linspace(0.05, 0.125, 5):
        trainer_ = BaseTrainer(
            train_ds_,
            test_ds_,
            local_desc_dim,
            global_desc_dim,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
            lambda_val=lambda_val,
        )
        trainer_.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument("--use_global", type=int, default=1)
    parser.add_argument(
        "--local_desc",
        type=str,
        default="r2d2",
    )
    parser.add_argument(
        "--local_desc_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="eigenplaces",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=2048,
    )
    args = parser.parse_args()

    run_function(
        args.dataset,
        args.local_desc,
        args.global_desc,
        int(args.local_desc_dim),
        int(args.global_desc_dim),
        bool(args.use_global),
    )
