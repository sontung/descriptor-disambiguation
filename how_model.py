from collections import OrderedDict

import cv2
import torch
import torchvision.transforms
from PIL import Image
import sys

sys.path.append("../how")
from how.networks import how_net

def _convert_checkpoint(state):
    """Enable loading checkpoints in the old format"""
    if "_version" not in state:
        # Old checkpoint format
        meta = state['meta']
        state['net_params'] = {
            "architecture": meta['architecture'],
            "pretrained": True,
            "skip_layer": meta['skip_layer'],
            "dim_reduction": {"dim": meta["dim"]},
            "smoothing": {"kernel_size": meta["feat_pool_k"]},
            "runtime": {
                "mean_std": [meta['mean'], meta['std']],
                "image_size": 1024,
                "features_num": 1000,
                "scales": [2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25],
                "training_scales": [1],
            },
        }

        state_dict = state['state_dict']
        state_dict['dim_reduction.weight'] = state_dict.pop("whiten.weight")
        state_dict['dim_reduction.bias'] = state_dict.pop("whiten.bias")

        state['_version'] = "how/2020"

    return state


class HowModel:
    def __init__(self):

        net_path = "../how/how_r50-.pth"

        # Load net
        state = _convert_checkpoint(torch.load(net_path, map_location='cpu'))
        device = "cuda"
        net = how_net.init_network(**state['net_params']).to(device)
        net.load_state_dict(state['state_dict'])
        net.eval()
        self.model = net
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (1024, 1024),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.conf = {"name": "how"}

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.model.forward_local(image, features_num=2000, scales=[2.0, 1.414, 1.0, 0.707, 0.5, 0.353, 0.25])
            desc, scores, kps, scales = output

        scales = 16/scales
        kps = kps.float().cpu().numpy()
        scales = scales.cpu().numpy()
        kps *= scales.reshape(-1, 1)

        # img = cv2.imread(name)
        # kps = kps.astype(int)
        # for u, v in kps:
        #     cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        # cv2.imwrite("debug/test.png", img)

        desc = desc.cpu().numpy()

        return kps, desc


if __name__ == '__main__':
    model = HowModel()
    model.process("/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/dusk/right/1424450252186877.jpg")
