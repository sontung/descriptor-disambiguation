import argparse
from pathlib import Path

import dd_utils
from dataset import SevenScenesDataset
from trainer import SevenScenesTrainer


def run_function(
    img_dir_,
    sfm_dir_,
    local_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = dd_utils.prepare_encoders(
        local_model, retrieval_model, global_desc_dim
    )

    scenes = ["chess", "redkitchen", "stairs", "fire", "heads", "office", "pumpkin"]

    results = {}
    for ds_name in scenes:
        print(f"Processing {ds_name}")
        train_ds_ = SevenScenesDataset(
            train=True,
            ds_name=ds_name,
            img_dir=f"{img_dir_}/7scenes_{ds_name}",
            sfm_model_dir=f"{sfm_dir_}/{ds_name}/sfm_gt",
        )
        test_ds_ = SevenScenesDataset(
            train=False,
            ds_name=ds_name,
            img_dir=f"{img_dir_}/7scenes_{ds_name}",
            sfm_model_dir=f"{sfm_dir_}/{ds_name}/sfm_gt",
        )

        set1 = set(train_ds_.img_ids)
        set2 = set(test_ds_.img_ids)
        set3 = set1.intersection(set2)
        assert len(set3) == 0

        trainer_ = SevenScenesTrainer(
            train_ds_,
            test_ds_,
            local_desc_dim,
            global_desc_dim,
            encoder,
            encoder_global,
            conf_ns,
            conf_ns_retrieval,
            using_global_descriptors,
        )
        err = trainer_.evaluate()
        print(f"    median translation error = {err[0]}")
        print(f"    median rotation error = {err[1]}")
        print(f"    percentage = {err[2]}")
        results[ds_name] = err
        del trainer_
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="../ace/datasets",
    )
    parser.add_argument(
        "--sfm_dir",
        type=str,
        default="../7scenes_reference_models",
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
    results = run_function(
        args.img_dir,
        args.sfm_dir,
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

    t0, r0, p0 = 0, 0, 0
    for ds in results:
        t_err, r_err, percent = results[ds]
        t0 += t_err
        r0 += r_err
        p0 += percent
        print(f"    {ds} {t_err:.3f}/{r_err:.3f} {percent:.3f}")
    print(f"    Avg. {t0/len(results):.3f}/{r0/len(results):.3f} {p0/len(results):.3f}")
