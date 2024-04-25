import argparse
from pathlib import Path
import numpy as np
import rerun as rr
import dd_utils
from dataset import CambridgeLandmarksDataset
from trainer import CambridgeLandmarksTrainer


def visualize(ds):
    rr.init("rerun_example_app")

    rr.connect()  # Connect to a remote viewer
    rr.spawn()  # Spawn a child process with a viewer and connect
    # rr.save("recording.rrd")  # Stream all logs to disk

    # Associate subsequent data with 42 on the “frame” timeline
    rr.set_time_sequence("frame", 42)

    # Log colored 3D points to the entity at `path/to/points`

    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ds.xyz_arr))
    cl, inlier_ind = point_cloud.remove_radius_outlier(nb_points=16, radius=5)
    rr.log(
        "path/to/points",
        rr.Points3D(ds.xyz_arr[inlier_ind], colors=ds.rgb_arr[inlier_ind] / 255),
    )


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

    # ds_name = "Cambridge_KingsCollege"
    ds_name = "Cambridge_GreatCourt"
    print(f"Processing {ds_name}")
    train_ds_ = CambridgeLandmarksDataset(
        train=True, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
    )
    test_ds_ = CambridgeLandmarksDataset(
        train=False, ds_name=ds_name, root_dir=f"{root_dir_}/{ds_name}"
    )
    visualize(train_ds_)

    set1 = set(train_ds_.rgb_files)
    set2 = set(test_ds_.rgb_files)
    set3 = set1.intersection(set2)
    assert len(set3) == 0

    trainer_ = CambridgeLandmarksTrainer(
        train_ds_,
        test_ds_,
        local_desc_dim,
        global_desc_dim,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        True,
    )
    trainer_2 = CambridgeLandmarksTrainer(
        train_ds_,
        test_ds_,
        local_desc_dim,
        global_desc_dim,
        encoder,
        encoder_global,
        conf_ns,
        conf_ns_retrieval,
        False,
    )
    trans, rot, name2err = trainer_.evaluate(return_name2err=True)
    trans2, rot2, name2err2 = trainer_2.evaluate(return_name2err=True)
    all_diff = {}
    all_name = []
    for name in name2err:
        e1 = name2err[name]
        e2 = name2err2[name]
        diff = e2 - e1
        all_diff[name] = diff
        all_name.append(name)
    n1 = min(all_name, key=lambda du1: all_diff[du1])
    n2 = max(all_name, key=lambda du1: all_diff[du1])
    print(n1, all_diff[n1], name2err[n1], name2err2[n1])
    print(n2, all_diff[n2], name2err[n2], name2err2[n2])
    print()


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
        default="mixvpr",
    )
    parser.add_argument(
        "--global_desc_dim",
        type=int,
        default=128,
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
