import torch
import torchvision.transforms
from PIL import Image
import sys


class DinoModel:
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.model.eval()
        self.model.cuda()
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
        self.conf = {"name": "dino"}

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image_descriptor = self.model(image.unsqueeze(0).cuda())[0]
        image_descriptor = image_descriptor.squeeze().cpu().numpy()  # 1024

        return image_descriptor
