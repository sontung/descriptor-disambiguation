import argparse

import dd_utils
from dataset import CMUDataset
from trainer import CMUTrainer

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def run_function(
    ds_dir,
    local_desc_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
    convert,
    lambda_val,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_desc_model, retrieval_model, global_desc_dim
    )
    if using_global_descriptors:
        print(f"Using {local_desc_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_desc_model}")
    results = []
    for slice_ in TEST_SLICES:
        print(f"Processing slice {slice_}")
        train_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}")
        test_ds_ = CMUDataset(ds_dir=f"{ds_dir}/slice{slice_}", train=False)

        trainer_ = CMUTrainer(
            train_ds_,
            test_ds_,
            local_desc_dim,
            global_desc_dim,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
            convert_to_db_desc=convert,
            lambda_val=lambda_val,
        )
        query_results = trainer_.evaluate()
        results.extend(query_results)
        trainer_.clear()
        del trainer_
        train_ds_.clear()

    if using_global_descriptors:
        result_file = open(
            f"output/cmu/CMU_eval_{local_desc_model}_{retrieval_model}_{global_desc_dim}_{convert}.txt",
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
    parser.add_argument("--convert", type=int, default=1)
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=0.3,
    )
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
        default="megaloc",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=8448,
    )
    args = parser.parse_args()

    run_function(
        args.dataset,
        args.local_desc,
        args.global_desc,
        int(args.local_desc_dim),
        int(args.global_desc_dim),
        bool(args.use_global),
        bool(args.convert),
        float(args.lambda_val),
    )
