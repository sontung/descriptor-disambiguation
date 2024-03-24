import argparse
from types import SimpleNamespace

import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from pathlib import Path

import dd_utils
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer
from mix_vpr_model import MVModel


def prepare_encoders(local_desc_model, retrieval_model, global_desc_dim):
    conf, default_conf = dd_utils.hloc_conf_for_all_models()
    model_dict = conf[local_desc_model]["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, model_dict["name"])
    encoder = Model(model_dict).eval().to(device)
    conf_ns = SimpleNamespace(**{**default_conf, **conf})
    conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
    conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]

    if retrieval_model == "mixvpr":
        encoder_global = MVModel(global_desc_dim)
        conf_ns_retrieval = None
    else:
        model_dict = conf[retrieval_model]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        if retrieval_model == "eigenplaces":
            model_dict.update(
                {
                    "variant": "EigenPlaces",
                    "backbone": "ResNet101",
                    "fc_output_dim": global_desc_dim,
                }
            )
        encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"][
            "resize_max"
        ]
    return encoder, conf_ns, encoder_global, conf_ns_retrieval


def run_function(
    root_dir_,
    local_model,
    retrieval_model,
    local_desc_dim,
    global_desc_dim,
    using_global_descriptors,
):
    encoder, conf_ns, encoder_global, conf_ns_retrieval = prepare_encoders(
        local_model, retrieval_model, global_desc_dim
    )
    folders = [
        item.name
        for item in Path(root_dir_).iterdir()
        if item.is_dir() and "Cambridge" in item.name
    ]

    results = {}
    for ds_name in folders:
        print(f"Processing {ds_name}")
        train_ds_ = CambridgeLandmarksDataset(
            train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )
        test_ds_ = CambridgeLandmarksDataset(
            train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
        )

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
        )
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
        default="../ace/datasets",
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
        default=4096,
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
        print(f"    {ds} {t_err:.3f}/{r_err:.3f}")
    print(f"    Avg. {t0/len(results):.3f}/{r0/len(results):.3f}")
