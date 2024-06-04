import argparse
from pathlib import Path
import numpy as np
import dd_utils
from clustering import reduce_map_using_min_cover
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer


def run_function(
    root_dir_,
    local_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_model, retrieval_model, global_desc_dim
    )
    if using_global_descriptors:
        print(f"Using {local_model} and {retrieval_model}-{global_desc_dim}")
    else:
        print(f"Using {local_model}")

    folders = [
        "GreatCourt",
        "KingsCollege",
        "OldHospital",
        "ShopFacade",
        "StMarysChurch",
    ]

    results = {}
    for ds_name in folders:
        # if ds_name != "GreatCourt":
        #     continue
        print(f"Processing {ds_name}")
        train_ds_ = CambridgeLandmarksDataset(
            train=True, ds_name=ds_name, root_dir=root_dir_
        )
        test_ds_ = CambridgeLandmarksDataset(
            train=False, ds_name=ds_name, root_dir=root_dir_
        )
        reduce_map_using_min_cover(train_ds_)

        # set1 = set(train_ds_.rgb_files)
        # set2 = set(test_ds_.rgb_files)
        # set3 = set1.intersection(set2)
        # assert len(set3) == 0

        trainer_ = CambridgeLandmarksTrainer(
            train_ds_,
            test_ds_,
            local_desc_dim,
            global_desc_dim,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
            lambda_val=0.5,
            convert_to_db_desc=False,
        )
        # continue
        err = trainer_.evaluate()
        print(f"    median translation error = {err[0]}")
        print(f"    median rotation error = {err[1]}")
        results[ds_name] = err
        del trainer_
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/cambridge",
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
        default="mixvpr",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=128,
    )
    args = parser.parse_args()
    results = run_function(
        args.dataset,
        args.local_desc,
        args.global_desc,
        int(args.local_desc_dim),
        int(args.global_desc_dim),
        bool(args.use_global),
    )

    if bool(args.use_global):
        print(
            f"Results: (translation/rotation) {args.local_desc} {args.global_desc}-{args.global_desc_dim}"
        )
    else:
        print(f"Results: (translation/rotation) {args.local_desc}")

    t0, r0 = 0, 0
    for ds in results:
        t_err, r_err = results[ds]
        t0 += t_err
        r0 += r_err
        print(f"    {ds} {t_err:.1f}/{r_err:.1f}")
    print(f"    Avg. {t0/len(results):.1f}/{r0/len(results):.1f}")
