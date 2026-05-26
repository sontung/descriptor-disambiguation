import torch
import torchvision.transforms
from PIL import Image
import sys
import torch.nn as nn

sys.path.append("/home/minhnxh/Documents/VinRobotic/VPR/salad")
from vpr_model import VPRModel as SaladVPR

def log_sinkhorn_iterations(
    Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([scores, bins], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class SaladModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        checkpoint = torch.load(
            "/home/minhnxh/Documents/VinRobotic/VPR/salad/weights/dino_salad.ckpt",
            map_location=self.device,
        )
        salad_model.load_state_dict(checkpoint)

        salad_model.eval()
        self.model = salad_model.to(self.device)
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
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    image_descriptor = self.model(image)
            else:
                image_descriptor = self.model(image)
        image_descriptor = image_descriptor.squeeze().float().cpu().numpy()  # 8448
        return image_descriptor

class SALAD_mask_agg(nn.Module):
    def __init__(
        self,
        num_channels=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        return f


if __name__ == '__main__':
    model = SaladModel()
    with torch.no_grad():
        out = model.process("/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/paper/overview.png")
    print()