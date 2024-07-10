import argparse

import main_aachen
import main_robotcar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset1",
        type=str,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default="/work/qvpr/data/raw/2020VisualLocalization/RobotCar-Seasons",
        help="Path to the dataset, default: %(default)s",
    )
    args = parser.parse_args()

    main_aachen.run_ablation_order(args.dataset1)
    main_robotcar.run_ablation_order(args.dataset2)
