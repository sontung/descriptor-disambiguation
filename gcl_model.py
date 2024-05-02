import argparse
import sys

import torch

sys.path.append("../generalized_contrastive_loss")
from PIL import Image
from src.factory import create_model as gcl_create_model
import torchvision.transforms


class TestParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        # self.parser.add_argument('--dataset', required=True, default='MSLS',
        #                          help='Name of the dataset [MSLS|7Scenes|TB_Places]')
        #
        # self.parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.parser.add_argument(
            "--subset", required=False, default="val", help="For MSLS. Subset to test"
        )

        self.parser.add_argument(
            "--query_idx_file", type=str, required=False, help="Query idx file, .json"
        )
        self.parser.add_argument(
            "--map_idx_file", type=str, required=False, help="Map idx file, .json"
        )

        self.parser.add_argument(
            "--backbone",
            type=str,
            default="vgg16",
            help="architectures: [vgg16, resnet18, resnet34, resnet50, resnet152, densenet161]",
        )
        self.parser.add_argument(
            "--pool", type=str, help="pool type|avg,max,GeM", default="GeM"
        )
        self.parser.add_argument(
            "--f_length", type=int, default=2048, help="feature length"
        )
        self.parser.add_argument(
            "--image_size",
            type=str,
            default="480,640",
            help="Input size, separated by commas",
        )
        self.parser.add_argument(
            "--norm", type=str, default="L2", help="Normalization descriptors"
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=16, help="Batch size"
        )

    def parse(self):
        self.opt = self.parser.parse_args()


class GCLModel:
    def __init__(self):
        self.conf = {"name": "gcl"}
        model_file = "../generalized_contrastive_loss/Models/Models/MSLS/MSLS_vgg16_GeM_480_GCL.pth"
        test_net = gcl_create_model(
            "vgg16", "GeM", norm="L2", mode="single"
        )
        test_net.load_state_dict(torch.load(model_file)["model_state_dict"])

        test_net.eval()
        test_net.cuda()
        self.model = test_net
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image = torchvision.transforms.functional.resize(image, (480, 640))
        with torch.no_grad():
            image_descriptor = self.model(image.unsqueeze(0).cuda())
        image_descriptor = image_descriptor.squeeze().cpu().numpy()  # 10752
        return image_descriptor


if __name__ == "__main__":
    du = GCLModel()
    g = du.process(
        "/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/dawn/rear/1418721355257224.jpg"
    )
    print(g.shape)
    print()
