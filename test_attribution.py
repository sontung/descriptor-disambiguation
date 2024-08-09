from tqdm import trange
import numpy as np
from captum.attr import GuidedGradCam, NeuronGradient

from collections import OrderedDict

import torch
import torchvision.transforms
from PIL import Image
import sys
import sys

sys.path.append("../salad")
from vpr_model import VPRModel as SaladVPR

salad_model = SaladVPR(
    backbone_arch="dinov2_vitb14",
    backbone_config={
        "num_trainable_blocks": 4,
        "return_token": True,
        "norm_layer": True,
    },
    agg_arch="SALAD",
    agg_config={
        "num_channels": 768,
        "num_clusters": 64,
        "cluster_dim": 128,
        "token_dim": 256,
    },
)
transform_op = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            (322, 322),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

checkpoint = torch.load("../salad/dino_salad.ckpt")
salad_model.load_state_dict(checkpoint)
salad_model = salad_model.cuda()
image = Image.open(
    "/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/night/right/1418235223631964.jpg"
).convert("RGB")
image = transform_op(image)
inp = image.unsqueeze(0).cuda()
# image_descriptor = model1.model(image.unsqueeze(0).cuda())
# image_descriptor = image_descriptor.squeeze().cpu().numpy()

neuron_deconv = GuidedGradCam(salad_model, salad_model.aggregator)
attribution = neuron_deconv.attribute(
    inp, 0, additional_forward_args=("allow_unused", True)
)

for k in salad_model.state_dict().keys():
    print(k)
print()
