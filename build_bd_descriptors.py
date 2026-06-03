import json
import glob
import sys

import cv2
import faiss
import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from pathlib import Path

import dd_utils
from dino_pipeline import PipelineConfig, DINOv2Pipeline


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
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

def get_patch_centres(image_ori, patch_size=14, proc_w=224, proc_h=224):

    ORIG_H, ORIG_W = image_ori.shape[:2]
    assert ORIG_H > 10 and ORIG_W > 10
    num_ph = proc_h // patch_size
    num_pw = proc_w // patch_size
    grid_y, grid_x = np.meshgrid(np.arange(num_ph), np.arange(num_pw), indexing="ij")

    cx = (grid_x.flatten() * patch_size + patch_size / 2) * (ORIG_W / proc_w)
    cy = (grid_y.flatten() * patch_size + patch_size / 2) * (ORIG_H / proc_h)

    return np.stack([cx, cy], axis=1).astype(np.float32)  # [P, 2]


def patch_mask_overlap(
    pixel_xy_topleft, masks, patch_size_x, patch_size_y, iou_thresh=0.3
):
    """
    pixel_xy_topleft : float32 [P, 2]  patch top-left corners (x, y)
    masks            : bool [N, H, W]
    Returns          : bool [P], float32 [P] max overlap ratio per patch
    """
    H, W = masks.shape[1], masks.shape[2]
    px0 = np.clip(pixel_xy_topleft[:, 0].astype(int), 0, W - 1)
    py0 = np.clip(pixel_xy_topleft[:, 1].astype(int), 0, H - 1)
    px1 = np.clip((pixel_xy_topleft[:, 0] + patch_size_x).astype(int), 0, W)
    py1 = np.clip((pixel_xy_topleft[:, 1] + patch_size_y).astype(int), 0, H)

    patch_area = patch_size_x * patch_size_y
    max_overlap = np.zeros(len(pixel_xy_topleft), dtype=np.float32)

    for mask in masks:
        for i in range(len(pixel_xy_topleft)):
            patch_region = mask[py0[i] : py1[i], px0[i] : px1[i]]
            inter = patch_region.sum()
            if inter == 0:
                continue
            max_overlap[i] = max(max_overlap[i], inter / patch_area)

    return max_overlap >= iou_thresh, max_overlap


def process_sam3_masks(masks, dino_patch_size=14, proc_w=224, proc_h=224):
    N, H_m, W_m = masks.shape  # native mask resolution

    mask_areas = np.sum(masks, axis=(1, 2))
    sorted_m_indices = np.argsort(mask_areas)

    processed_pixels = np.zeros((H_m, W_m), dtype=bool)
    observations = []
    m_indices = []
    for m_idx in sorted_m_indices:
        current_mask = masks[m_idx] > 0
        unique_mask_region = current_mask & ~processed_pixels

        original_pixel_count = mask_areas[m_idx]
        if original_pixel_count == 0:
            continue

        unique_pixel_count = np.sum(unique_mask_region)
        if (unique_pixel_count / original_pixel_count) < 0.5:
            continue

        processed_pixels |= current_mask

        ORIG_H, ORIG_W = unique_mask_region.shape[:2]
        patch_size_x = dino_patch_size * (ORIG_W / proc_w)
        patch_size_y = dino_patch_size * (ORIG_H / proc_h)

        # get_patch_centres returns centres — shift to top-left for bbox IoU
        pixel_xy = get_patch_centres(
            unique_mask_region, patch_size=dino_patch_size, proc_w=proc_w, proc_h=proc_h
        )
        pixel_xy_topleft = pixel_xy - np.array([patch_size_x / 2, patch_size_y / 2])

        inside, overlap_scores = patch_mask_overlap(
            pixel_xy_topleft,
            unique_mask_region[None],
            patch_size_x,
            patch_size_y,
            iou_thresh=0.2,
        )

        # obs = {
        #     "mask": unique_mask_region.astype(float)*255,
        #     # "chosen_patches": inside.astype(int).tolist(),
        # }
        observations.append(unique_mask_region.astype(float) * 255)
        m_indices.append(m_idx)

    return observations, m_indices

def find_dino_patch_coords(
    unique_mask_region, dino_patch_size=14, proc_w=224, proc_h=224
):
    ORIG_H, ORIG_W = unique_mask_region.shape[:2]
    patch_size_x = dino_patch_size * (ORIG_W / proc_w)
    patch_size_y = dino_patch_size * (ORIG_H / proc_h)

    # get_patch_centres returns centres — shift to top-left for bbox IoU
    pixel_xy = get_patch_centres(
        unique_mask_region, patch_size=dino_patch_size, proc_w=proc_w, proc_h=proc_h
    )
    pixel_xy_topleft = pixel_xy - np.array([patch_size_x / 2, patch_size_y / 2])

    inside, overlap_scores = patch_mask_overlap(
        pixel_xy_topleft,
        unique_mask_region[None],
        patch_size_x,
        patch_size_y,
        iou_thresh=0.2,
    )
    return inside, pixel_xy_topleft[inside]


def visualize_stem_masks(
    image, masks, patches, dino_patch_size=14, proc_w=224, proc_h=224
):

    # Create a deep copy to prevent mutating the original input array in place
    vis = image.copy()
    ORIG_H, ORIG_W = vis.shape[:2]

    # Calculate dynamic patch scaling ratios
    patch_size_x = dino_patch_size * (ORIG_W / proc_w)
    patch_size_y = dino_patch_size * (ORIG_H / proc_h)

    # 1. Draw masks as a semi-transparent overlay with random colors
    if masks is not None and len(masks) > 0:
        overlay = vis.copy()

        for mask in masks:
            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue

            # Generate a random vibrant BGR color (avoiding completely dark colors)
            color_array = np.random.randint(50, 256, size=3, dtype=np.uint8)
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))

            # Blend the random color onto the overlay canvas copy
            overlay[mask_bool] = (vis[mask_bool] * 0.4 + color_array * 0.6).astype(
                np.uint8
            )

            # Draw contours using the matching color for crisp structural boundaries
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 1)

        vis = overlay

    # 2. Draw selected bounding patch boxes
    if patches is not None:
        for pixel_xy in patches:
            for cx, cy in pixel_xy.astype(int):
                px0, py0 = cx, cy
                px1 = int(cx + patch_size_x)
                py1 = int(cy + patch_size_y)
                # Clip values internally to safely handle edge boundaries near borders
                px1 = min(px1, ORIG_W)
                py1 = min(py1, ORIG_H)
                cv2.rectangle(vis, (px0, py0), (px1, py1), (220, 0, 0), 10)

    return vis

def mask_assignment(image_ori, dino_patches):
    # PIL image.size returns (width, height)
    ORIG_W, ORIG_H = image_ori.size

    # Generate pixel grid coordinates vectorially
    y, x = np.indices((ORIG_W, ORIG_H))
    target_pix_np = np.stack([x.ravel(), y.ravel()], axis=1)
    target_pix_np[:, [0, 1]] = target_pix_np[:, [1, 0]]
    # Initialize assignments with -1
    # mask_keypoint_assignments = np.full(target_pix_np.shape[0], -1, dtype=int)
    assignment_map = np.full((ORIG_W, ORIG_H), -1, dtype=np.int16)

    if dino_patches is not None and len(dino_patches) > 0:
        BASE_PATCH_SIZE = 14

        # Calculate dynamic patch dimensions
        patch_size_x = int(BASE_PATCH_SIZE * (ORIG_W / 224.0))
        patch_size_y = int(BASE_PATCH_SIZE * (ORIG_H / 224.0))

        # 1. Pre-convert dino_patches to standard NumPy arrays on CPU
        processed_masks = []
        for patch_coords in dino_patches:
            if len(patch_coords) == 0:
                processed_masks.append(np.empty((0, 2)))
                continue
            if torch.is_tensor(patch_coords):
                processed_masks.append(patch_coords.detach().cpu().numpy())
            else:
                processed_masks.append(np.array(patch_coords))

        # 2. Vectorized assignment loop per mask layer (instead of per pixel)
        # We loop over masks backward so earlier masks take priority
        # (matching the `break` behavior of your original code)
        for mask_idx in range(len(processed_masks) - 1, -1, -1):
            patch_coords = processed_masks[mask_idx]
            if patch_coords.size == 0:
                continue

            # Iterate through the patches for this mask
            for pt in patch_coords:
                px0, py0 = max(0, int(pt[0])), max(0, int(pt[1]))
                assignment_map[px0:px0 + patch_size_x, py0:py0 + patch_size_y] = mask_idx

            # # Extract top-left bounds
            # px0 = patch_coords[:, 0]  # Shape: (Num_Patches,)
            # py0 = patch_coords[:, 1]  # Shape: (Num_Patches,)
            #
            # # Use broadcasting to check all pixels against all patches for this mask simultaneously
            # # target_pix_np[:, 0] is X (shape: Pixels, 1)
            # # px0 is patch X bounds (shape: 1, Num_Patches)
            # x_in_patch = (target_pix_np[:, [0]] >= px0) & (target_pix_np[:, [0]] < px0 + patch_size_x)
            # y_in_patch = (target_pix_np[:, [1]] >= py0) & (target_pix_np[:, [1]] < py0 + patch_size_y)
            #
            # # A pixel is in a patch if both X and Y conditions are met
            # in_any_patch = np.any(x_in_patch & y_in_patch, axis=1)
            #
            # # Assign mask index to valid pixels
            # mask_keypoint_assignments[in_any_patch] = mask_idx

    # 4. Reconstruct the precise structural output matching your original script
    # This keeps downstream code entirely unchanged!
    y_indices, x_indices = np.indices((ORIG_W, ORIG_H))

    # Flatten everything to a 1D sequence to match your expected layout
    target_pix_np2 = np.stack([x_indices.ravel(), y_indices.ravel()], axis=1)
    target_pix_np2[:, [0, 1]] = target_pix_np2[:, [1, 0]]

    mask_keypoint_assignments2 = assignment_map.ravel()
    return mask_keypoint_assignments2, target_pix_np2


def main():
    cfg = PipelineConfig(
        model_name="facebook/dinov2-base", batch_size=4, return_patch_tokens=True
    )
    dino_pipeline = DINOv2Pipeline(cfg)

    mask_agg = SALAD_mask_agg(num_clusters=16, cluster_dim=16).to("cuda")
    mask_agg.load_state_dict(
        torch.load("/home/vr/work/sign_based_loc/mask_aggregator_best.pth", map_location="cuda")
    )
    mask_agg.eval()

    ds_dir = "/home/vr/work/datasets/aachen10/images_upright"

    edge_file = "/home/vr/work/sign_based_loc/outputs/edges.json"
    obs_file = "/home/vr/work/sign_based_loc/outputs/observations.json"
    sam_feats_h5 = h5py.File("/home/vr/work/sign_based_loc/outputs/sam3_results_db.h5", "r")
    with open(edge_file) as f:
        edges = [tuple(e) for e in json.load(f)]
    with open(obs_file) as f:
        observations = json.load(f)
    unique_obs_id = np.unique(edges)
    im2obs = {}
    for idx in unique_obs_id:
        obs = observations[idx]
        name = obs["stem"]
        im2obs.setdefault(name, []).append(obs)

    all_images = glob.glob(f"{ds_dir}/**/*.jpg", recursive=True)
    print(len(all_images))

    all_db_desc = []
    all_query_desc = []
    global_row_index = 0
    row_index_query = 0
    meta_h5 = h5py.File("output/db_metadata_mapping.h5", "w")
    query_meta_h5 = h5py.File("output/query_metadata_mapping.h5", "w")

    for img_name in tqdm(all_images):

        stem = Path(img_name).stem
        if stem not in sam_feats_h5:
            continue
        all_masks = sam_feats_h5[stem]["architecture"]["masks"][:]
        if "db" in img_name:
            if stem not in im2obs:
                continue
            db_image = True
            all_observations = im2obs[stem]
            mask_indices = [obs["mask_index"] for obs in all_observations]
            all_masks = all_masks[mask_indices]
        else:
            db_image = False

        all_observations, m_indices2 = process_sam3_masks(all_masks)
        chosen_patches = [find_dino_patch_coords(du)[0] for du in all_observations]
        dino_patches = [find_dino_patch_coords(du)[1] for du in all_observations]
        counts = np.array([np.sum(cp) for cp in chosen_patches])
        image_ori = Image.open(img_name)
        with torch.no_grad():
            res = dino_pipeline.run(
                [image_ori],
            )
            dino_descriptors = res["patches"][0].reshape(-1, 768)

            mask_feats = []
            for cp in chosen_patches:
                mask_feats.append(
                    (torch.from_numpy(dino_descriptors) * cp[:, None])
                    .float()
                    .T.reshape(768, 16, 16)
                )
            bd_descriptors = mask_agg(torch.stack(mask_feats).cuda())
            bd_descriptors = bd_descriptors.cpu()

            mask_keypoint_assignments, target_pix_np = mask_assignment(image_ori, dino_patches)

            num_descriptors = bd_descriptors.shape[0]
            if db_image:
                for i in range(num_descriptors):
                    mask0 = mask_keypoint_assignments == i
                    if np.sum(mask0) <= 100:
                        continue
                    all_db_desc.append(bd_descriptors[i])
                    grp = meta_h5.create_group(str(global_row_index))
                    grp.create_dataset("image_name", data=img_name)
                    grp.create_dataset("target_pixel", data=mask0, compression="gzip", compression_opts=4)

                    global_row_index += 1

            else:

                for i in range(num_descriptors):
                    mask0 = mask_keypoint_assignments == i
                    if np.sum(mask0) <= 100:
                        continue
                    all_query_desc.append(bd_descriptors[i])
                    grp = query_meta_h5.create_group(str(row_index_query))
                    grp.create_dataset("image_name", data=img_name)
                    grp.create_dataset("target_pixel", data=mask0, compression="gzip", compression_opts=4)

                    row_index_query += 1

            # vis_img = np.array(Image.open(img_name))
            # overlay = vis_img.copy()
            # max_mask_id = np.max(mask_keypoint_assignments)
            # num_colors = max_mask_id + 1
            # colors = []
            # for i in range(num_colors):
            #     hue = int(180 * i / num_colors)  # OpenCV hue range is 0-179
            #     hsv_color = np.uint8([[[hue, 255, 255]]])
            #     bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            #     colors.append(tuple(int(c) for c in bgr_color))
            #
            # # 3. Filter out unassigned pixels (-1) to draw efficiently
            # assigned_indices = np.where(mask_keypoint_assignments != -1)[0]
            #
            # # 2. Draw the mask points onto the OVERLAY image
            # for idx in assigned_indices:
            #     x, y = target_pix_np[idx]
            #     mask_id = mask_keypoint_assignments[idx]
            #     color = colors[mask_id]
            #
            #     # Using a radius of 1 or 2 fills the patch space nicely without creating giant overlapping blobs
            #     cv2.circle(overlay, (int(x), int(y)), radius=2, color=color, thickness=-1)
            #
            # # 3. Blend the original image and the overlay together
            # # alpha controls original image visibility, beta controls mask transparency
            # alpha = 0.6  # Importance of original image
            # beta = 0.4  # Importance of mask colors
            # gamma = 0  # Scalar added to each sum
            #
            # vis_img = cv2.addWeighted(vis_img, alpha, overlay, beta, gamma)
            #
            # # vis_img = np.array(Image.open(img_name))
            # vis_img = visualize_stem_masks(vis_img, None, dino_patches)
            # cv2.imwrite(f"debug/test{stem}.jpg", vis_img)

    meta_h5.close()
    sam_feats_h5.close()
    query_meta_h5.close()
    # Save both assets
    torch.save(torch.stack(all_db_desc), "output/all_db_descriptors_matrix.pt")
    torch.save(torch.stack(all_query_desc), "output/all_query_descriptors_matrix.pt")
    return


def draw_masked_kps(img_name, mask):
    image_ori = Image.open(img_name)
    ORIG_W, ORIG_H = image_ori.size
    y_indices, x_indices = np.indices((ORIG_W, ORIG_H))

    # Flatten everything to a 1D sequence to match your expected layout
    target_pix_np2 = np.stack([x_indices.ravel(), y_indices.ravel()], axis=1)
    target_pix_np2[:, [0, 1]] = target_pix_np2[:, [1, 0]]
    vis_img = np.array(Image.open(img_name))
    overlay = vis_img.copy()


    # 2. Draw the mask points onto the OVERLAY image
    for x, y in target_pix_np2[mask]:

        # Using a radius of 1 or 2 fills the patch space nicely without creating giant overlapping blobs
        cv2.circle(overlay, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)

    # 3. Blend the original image and the overlay together
    # alpha controls original image visibility, beta controls mask transparency
    alpha = 0.6  # Importance of original image
    beta = 0.4  # Importance of mask colors
    gamma = 0  # Scalar added to each sum

    vis_img = cv2.addWeighted(vis_img, alpha, overlay, beta, gamma)
    return vis_img


def evaluate():
    db_meta = h5py.File("/home/vr/work/descriptor-disambiguation/output/db_metadata_mapping.h5", "r")
    db_desc = torch.load("/home/vr/work/descriptor-disambiguation/output/all_db_descriptors_matrix.pt")
    db_desc = db_desc.numpy()

    query_meta = h5py.File("/home/vr/work/descriptor-disambiguation/output/query_metadata_mapping.h5", "r")
    query_desc = torch.load("/home/vr/work/descriptor-disambiguation/output/all_query_descriptors_matrix.pt")
    query_desc = query_desc.numpy()

    index2 = faiss.IndexFlatL2(256)  # build the index
    res2 = faiss.StandardGpuResources()
    gpu_index_flat_for_image_desc = faiss.index_cpu_to_gpu(res2, 0, index2)
    gpu_index_flat_for_image_desc.add(db_desc.astype(np.float32))
    _, retrieved_indices = gpu_index_flat_for_image_desc.search(
        query_desc.astype(np.float32), 1
    )
    retrieved_indices = retrieved_indices.flatten()
    idx0 = 0
    for query_idx, db_idx in enumerate(retrieved_indices):
        m1 = query_meta[str(query_idx)]
        m2 = db_meta[str(db_idx)]

        im1 = draw_masked_kps(m1["image_name"][()].decode('utf-8'), m1["target_pixel"][:])
        im2 = draw_masked_kps(m2["image_name"][()].decode('utf-8'), m2["target_pixel"][:])
        vis2 = dd_utils.concat_images_different_sizes([im1, im2])
        idx0 += 1
        cv2.imwrite(f"debug/{str(idx0)}.jpg", cv2.cvtColor(vis2, cv2.COLOR_RGB2BGR))
    return

if __name__ == '__main__':
    # main()
    evaluate()
