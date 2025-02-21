import torch
import torchvision.transforms
from PIL import Image
import sys

sys.path.append("../salad")
from vpr_model import VPRModel as SaladVPR


class SaladModel:
    def __init__(self):
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

        checkpoint = torch.load("../salad/dino_salad.ckpt")
        salad_model.load_state_dict(checkpoint)

        salad_model.eval()
        salad_model.cuda()
        self.model = salad_model
        self.transform = torchvision.transforms.Compose(
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
        self.conf = {"name": "salad"}

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image_descriptor = self.model(image.unsqueeze(0).cuda())
        image_descriptor = image_descriptor.squeeze().cpu().numpy()  # 8448
        return image_descriptor


if __name__ == '__main__':
    model = SaladModel()
    with torch.no_grad():
        out = model.process("/home/n11373598/work/descriptor-disambiguation/datasets/robotcar/images/overcast-reference/rear/1417178903018872.jpg")
    print()