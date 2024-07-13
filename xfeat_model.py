from collections import OrderedDict

import cv2
import torch
import sys
import imageio

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


class XfeatModel:
    def __init__(self):
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

        self.conf = {"name": "xfeat"}

    def process(self, name):
        image = imageio.v2.imread(name)
        image = self.model.parse_input(image)

        # out1 = self.model.detectAndCompute(image, top_k=4096)[0]
        out1 = self.model.detectAndComputeDense(image, top_k=None)
        kps = out1["keypoints"].cpu().numpy()[0]
        desc = out1["descriptors"].cpu().numpy()[0]

        # img = cv2.imread(name)
        # kps = kps.astype(int)
        # for u, v in kps:
        #     cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        # cv2.imwrite("debug/test.png", img)

        return kps, desc


if __name__ == '__main__':
    model = XfeatModel()
    model.process("/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/dusk/right/1424450252186877.jpg")
