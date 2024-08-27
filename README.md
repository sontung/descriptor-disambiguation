# FUSELOC: Fusing Global and Local Descriptors to Disambiguate 2D-3D Matching in Visual Localization
[[Arxiv]](https://arxiv.org/abs/2408.12037)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

To get started with FUSELOC, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/FUSELOC.git
cd FUSELOC
pip install -r requirements.txt
```

## Usage

Assume the dataset is downloaded somewhere, here is an example of usage:
```bash
python main_robotcar.py --dataset /work/qvpr/data/raw/2020VisualLocalization/RobotCar-Seasons --local_desc "d2net" --local_desc_dim 512 --global_desc "salad" --global_desc_dim 8448 --use_global 1 --convert 1
```
