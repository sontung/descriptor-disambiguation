import argparse

import numpy as np

import dd_utils
from dataset import RobotCarDataset
from trainer import RobotCarTrainer

ABLATION_METHODS = [
    ["salad", 8448],
    ["crica", 10752],
    ["eigenplaces", 2048],
    ["mixvpr", 4096],
]


def run_ablation(ds_dir):
    using_global_descriptors = True
    train_ds_ = RobotCarDataset(ds_dir=ds_dir)
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=True)
    local_desc_model = "d2net"
    for retrieval_model, global_desc_dim in ABLATION_METHODS:
        encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
            local_desc_model, retrieval_model, global_desc_dim
        )

        print(f"Using {local_desc_model} and {retrieval_model}-{global_desc_dim}")

        for lambda_val in np.linspace(0, 1, 11):
            if lambda_val == 0.0:
                continue
            trainer_ = RobotCarTrainer(
                train_ds_,
                test_ds_,
                512,
                global_desc_dim,
                encoder,
                encoder_global,
                conf_ns,
                conf_ns_retrieval,
                using_global_descriptors,
                lambda_val=lambda_val,
                convert_to_db_desc=True,
            )
            trainer_.evaluate()


def run_function(
    ds_dir,
    local_desc_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
    convert,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_desc_model, retrieval_model, global_desc_dim
    )
    if using_global_descriptors:
        print(f"Using {local_desc_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_desc_model}")
    train_ds_ = RobotCarDataset(ds_dir=ds_dir)
    test_ds_ = RobotCarDataset(ds_dir=ds_dir, train=False, evaluate=True)

    # for lambda_val in [0.3]:
    for lambda_val in np.linspace(0, 1, 11):
        if lambda_val == 0.0:
            continue
        trainer_ = RobotCarTrainer(
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
            convert_to_db_desc=convert,
        )
        trainer_.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument("--use_global", type=int, default=1)
    parser.add_argument("--convert", type=int, default=1)

    parser.add_argument(
        "--local_desc",
        type=str,
        default="d2net",
    )
    parser.add_argument(
        "--local_desc_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--global_desc",
        type=str,
        default="salad",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=8448,
    )

    args = parser.parse_args()

    run_ablation(args.dataset)

    # run_function(
    #     args.dataset,
    #     args.local_desc,
    #     args.global_desc,
    #     int(args.local_desc_dim),
    #     int(args.global_desc_dim),
    #     bool(args.use_global),
    #     bool(args.convert),
    # )
