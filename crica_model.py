import sys
sys.path.append("../CricaVPR")

import torch
from PIL import Image
from collections import OrderedDict
import torchvision.transforms
import network as crica_net_lib


class CricaModel:
    def __init__(self):
        self.conf = {"name": "crica"}

        model = crica_net_lib.CricaVPRNet()
        checkpoint = torch.load("../CricaVPR/CricaVPR.pth")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module'):
            state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        model.eval()
        self.model = model
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process(self, name):
        image = Image.open(name).convert("RGB")
        image = self.transform(image)
        image = torchvision.transforms.functional.resize(image, (224, 224))
        image_descriptor = self.model(image.unsqueeze(0).cuda())
        image_descriptor = image_descriptor.squeeze().cpu().numpy()
        return image_descriptor
