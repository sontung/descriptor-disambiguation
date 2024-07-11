import argparse

import main_aachen
import main_robotcar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="aachen",
        help="Path to the dataset, default: %(default)s",
    )
    args = parser.parse_args()
    if args.benchmark == "aachen":
        main_aachen.run_ablation_order(args.dataset)
    elif args.benchmark == "robotcar":
        main_robotcar.run_ablation_order(args.dataset)
    else:
        raise NotImplementedError
