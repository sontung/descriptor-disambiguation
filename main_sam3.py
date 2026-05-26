import glob
from pathlib import Path
import h5py
import numpy as np
import torch
import os


import ace_util
from salad_model import SALAD_mask_agg
import numpy as np
import torch

def get_patch_centres(image_ori, patch_size=14, proc_w=224, proc_h=224):
    ORIG_H, ORIG_W = image_ori.shape[:2]
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

    return observations
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




# def main():
#     sam_feats_h5 = h5py.File(
#         "/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/sam3_results_db.h5", "r")
#     sfm_model_dir = "/home/minhnxh/Documents/VinRobotic/aachen10/3D-models/aachen_cvpr2018_db.nvm"
#     dino_feat = h5py.File(
#         "/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/dino_features.h5", "r")
#     # cfg = /home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/mask_aggregator_best.pth
#
#     # Load NVM
#     (
#         xyz_arr, image2points, image2name,
#         image2pose, image2info, image2uvs, rgb_arr
#     ) = ace_util.read_nvm_file(sfm_model_dir)
#     name2image = {v: k for k, v in image2name.items()}
#
#     mask_agg = SALAD_mask_agg(num_clusters=16, cluster_dim=16).to("cuda")
#
#     mask_agg.load_state_dict(
#         torch.load(
#             "/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/mask_aggregator_best.pth",
#             map_location="cuda")
#     )
#     mask_agg.eval()
#
#     obslist = []
#     for name in image2name.values():
#         all_masks = sam_feats_h5[Path(name).stem]["architecture"][
#             "masks"
#         ][:]
#         all_observations = process_sam3_masks(all_masks)
#         obslist.append(all_observations)
#         dino_patches = [find_dino_patch_coords(du)[1] for du in all_masks]
#         chosen_patches = [find_dino_patch_coords(du)[0] for du in all_masks]
#
#         dino_descriptors = dino_feat[Path(name).stem]["dino_features"][:].reshape(-1, 768)
#         mask_feats = []
#
#         for cp in chosen_patches:
#             mask_feats.append(
#                 (torch.from_numpy(dino_descriptors) * torch.from_numpy(cp).int()[:, None])
#                 .float()
#                 .T.reshape(768, 16, 16)
#             )
#         bd_descriptors = mask_agg(torch.stack(mask_feats).cuda())
#
#         globaldesc = bd_descriptors
#         maskls = []
#         bd_indices = []
#
#         for i in range(all_masks.shape[0]):
#             masked_kps = np.array(np.where(all_masks[i])).T
#             maskls.append(masked_kps)
#             bd_indices.append(np.array([i] * masked_kps.shape[0]))
#
#         maskls = np.vstack(maskls)
#         bd_indices = np.hstack(bd_indices)
#
#         # globalnmask = {'array1': bd_indices, 'array2': maskls, 'array3': globaldesc}
#         globalnmask = {'array1': bd_indices, 'array2': maskls, 'array3': globaldesc.detach().cpu().numpy()}
#         with h5py.File("descriptor_mask_test.h5", "a") as f:
#             for name, (key, data) in zip(image2name.values(), globalnmask.items()):
#                 f.create_dataset(name, data=data)
#
#         break



def main():
    sam_feats_h5 = h5py.File(
        "/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/sam3_results_db.h5", "r")
    sfm_model_dir = "/home/minhnxh/Documents/VinRobotic/aachen10/3D-models/aachen_cvpr2018_db.nvm"
    dino_feat = h5py.File("/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/dino_features.h5", "r")
    # cfg = /home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/mask_aggregator_best.pth

    # Load NVM
    (
        xyz_arr, image2points, image2name,
        image2pose, image2info, image2uvs, rgb_arr
    ) = ace_util.read_nvm_file(sfm_model_dir)
    name2image = {v: k for k, v in image2name.items()}

    mask_agg = SALAD_mask_agg(num_clusters=16, cluster_dim=16).to("cuda")

    mask_agg.load_state_dict(
        torch.load("/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information/mask_aggregator_best.pth", map_location="cuda")
    )
    mask_agg.eval()

    all_image_names = glob.glob("/home/minhnxh/Documents/VinRobotic/aachen10/images_upright/**/*.jpg", recursive=True)

    with h5py.File("descriptor_mask.h5", "a") as f:  # ← Open file once

        for name in all_image_names:
            stem = Path(name).stem
            print(f"Processing: {stem}")  # ← Added for progress

            if stem in f:
                continue

            if stem not in sam_feats_h5:
                continue

            all_masks = sam_feats_h5[Path(name).stem]["architecture"]["masks"][:]
            chosen_patches = [find_dino_patch_coords(du)[0] for du in all_masks]

            dino_descriptors = dino_feat[stem]["dino_features"][:].reshape(-1, 768)
            mask_feats = []

            for cp in chosen_patches:
                mask_feats.append(
                    (torch.from_numpy(dino_descriptors) * torch.from_numpy(cp).int()[:, None])
                    .float()
                    .T.reshape(768, 16, 16)
                )

            bd_descriptors = mask_agg(torch.stack(mask_feats).cuda())
            globaldesc = bd_descriptors

            # Build maskls and bd_indices
            maskls = []
            bd_indices = []
            for i in range(all_masks.shape[0]):
                masked_kps = np.array(np.where(all_masks[i])).T
                maskls.append(masked_kps)
                bd_indices.append(np.array([i] * masked_kps.shape[0]))

            maskls = np.vstack(maskls)
            bd_indices = np.hstack(bd_indices)



            # === FIXED HDF5 WRITING ===
            img_group = f.create_group(stem)  # One group per image

            img_group.create_dataset('bd_indices', data=bd_indices, compression="gzip")
            img_group.create_dataset('maskls', data=maskls, compression="gzip")
            img_group.create_dataset('globaldesc',
                                     data=globaldesc.detach().cpu().numpy(),
                                     compression="gzip")

            # Optional: clean up memory
            del globaldesc, mask_feats, bd_descriptors, maskls, bd_indices
            torch.cuda.empty_cache()

            f.flush()


if __name__ == '__main__':
    main()
