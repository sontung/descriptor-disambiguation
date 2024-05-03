import argparse
import glob
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
)
from hloc.pipelines.RobotCar import colmap_from_nvm


def generate_query_list(dataset, image_dir, path):
    h, w = 1024, 1024
    intrinsics_filename = "intrinsics/{}_intrinsics.txt"
    cameras = {}
    for side in ["left", "right", "rear"]:
        with open(dataset / intrinsics_filename.format(side), "r") as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ["SIMPLE_RADIAL", w, h, fx, cx, cy, 0.0]
            cameras[side] = [str(p) for p in params]

    queries = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    queries = [
        Path(q).relative_to(image_dir.parents[0]).as_posix() for q in sorted(queries)
    ]

    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images/"

    outputs = args.outputs  # where everything will be saved
    outputs.mkdir(exist_ok=True, parents=True)
    sift_sfm = outputs / "sfm_sift"

    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["eigenplaces"]
    feature_conf = extract_features.confs["d2net-ss"]
    matcher_conf = match_features.confs["NN-ratio"]

    # extract_features.main(
    #     extract_features.confs["d2net-ss"],
    #     images,
    #     Path(
    #         "/home/n11373598/hpc-home/work/descriptor-disambiguation/outputs/robotcar"
    #     ),
    # )

    colmap_from_nvm.main(
        dataset / "3D-models/all-merged/all.nvm",
        dataset / "3D-models/overcast-reference.db",
        sift_sfm,
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        db_model=sift_sfm,
    )

    features = extract_features.main(feature_conf, images, outputs)

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/robotcar",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
