# Copyright © Niantic, Inc. 2022.
import json
import random
from distutils.util import strtobool

import PIL
import cv2
import numpy as np
import poselib
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing="ij")
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output


def read_nvm_file(file_name, return_rgb=False):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    nb_cameras = int(lines[2])
    image2info = {}
    image2pose = {}
    image2name = {}
    unique_names = []
    for i in tqdm(range(nb_cameras), desc="Reading cameras"):
        cam_info = lines[3 + i]
        if "\t" in cam_info:
            img_name, info = cam_info.split("\t")
            focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = map(float, info.split(" "))
        else:
            img_name, focal, qw, qx, qy, qz, tx, ty, tz, radial, _ = cam_info.split(" ")
            focal, qw, qx, qy, qz, tx, ty, tz, radial = map(
                float, [focal, qw, qx, qy, qz, tx, ty, tz, radial]
            )
        image2name[i] = img_name
        assert img_name not in unique_names
        unique_names.append(img_name)
        image2info[i] = [focal, radial]
        image2pose[i] = [qw, qx, qy, qz, tx, ty, tz]
    nb_points = int(lines[4 + nb_cameras])
    image2points = {}
    image2uvs = {}
    xyz_arr = np.zeros((nb_points, 3), np.float64)
    rgb_arr = np.zeros((nb_points, 3), np.uint8)
    for j in tqdm(range(nb_points), desc="Reading points"):
        point_info = lines[5 + nb_cameras + j].split(" ")
        x, y, z, r, g, b, nb_features = map(float, point_info[:7])
        xyz_arr[j] = [x, y, z]
        rgb_arr[j] = [r, g, b]
        features_info = point_info[7:]
        nb_features = int(nb_features)
        for k in range(nb_features):
            image_id, feature_id, u, v = features_info[k * 4 : (k + 1) * 4]
            image_id, feature_id = map(int, [image_id, feature_id])
            u, v = map(float, [u, v])
            image2points.setdefault(image_id, []).append(j)
            image2uvs.setdefault(image_id, []).append([u, v])

    # for image_id in image2uvs:
    #     img_name = image2name[image_id]
    #     image = cv2.imread(f"datasets/Cambridge_KingsCollege/{img_name.split('.jpg')[0]}.png")
    #     uvs = np.array(image2uvs[image_id])
    #     pids = np.array(image2points[image_id])
    #     xyz = xyz_arr[pids]
    #     focal, radial = image2info[image_id]
    #     r2 = np.sqrt(np.sum(np.square(uvs), 1))*radial
    #     r2 = np.expand_dims(r2, 1)
    #     uvs_undistorted = uvs*(1+r2)
    #     print()
    if return_rgb:
        return xyz_arr, image2points, image2name, rgb_arr
    else:
        return xyz_arr, image2points, image2name


def return_pose_mat(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    # convert world->cam to cam->world for evaluation
    pose_4x4_inv = np.linalg.inv(pose_4x4)
    return pose_4x4_inv


def read_reconstruction_mapillary(a_file):
    f = open(a_file, "r")
    data = json.load(f)
    f.close()
    print(len(data))
    return


def localize_pose_lib(pairs, f, c1, c2, max_error=16.0):
    """
    using pose lib to compute (usually best)
    """
    camera = {
        "model": "SIMPLE_PINHOLE",
        "height": int(c1 * 2),
        "width": int(c2 * 2),
        "params": [f, c1, c2],
    }
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        xyz = np.array(xyz).reshape((3, 1))
        xy = np.array(xy)
        xy = xy.reshape((2, 1)).astype(np.float64)
        image_points.append(xy)
        object_points.append(xyz)
    pose, info = poselib.estimate_absolute_pose(
        image_points, object_points, camera, {"max_reproj_error": max_error}, {}
    )
    return pose, info


def localize_pose_lib_light(pairs, f, c1, c2, max_error=16.0):
    """
    using pose lib to compute (usually best)
    """
    camera = {
        "model": "SIMPLE_PINHOLE",
        "height": int(c1 * 2),
        "width": int(c2 * 2),
        "params": [f, c1, c2],
    }
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        xyz = np.array(xyz).reshape((3, 1))
        xy = np.array(xy)
        xy = xy.reshape((2, 1)).astype(np.float64)
        image_points.append(xy)
        object_points.append(xyz)
    pose, info = poselib.estimate_absolute_pose(
        image_points,
        object_points,
        camera,
        {"max_reproj_error": max_error, "max_iterations": 2000},
        {},
    )
    return pose, info


def find_oob(example, uv):
    w, h = example[1].shape[:2]

    oob_mask1 = np.bitwise_and(0 <= uv[:, 0], uv[:, 0] < h)
    oob_mask2 = np.bitwise_and(0 <= uv[:, 1], uv[:, 1] < w)
    oob_mask = np.bitwise_and(oob_mask1, oob_mask2)
    return oob_mask


def normalize(a, b, arr, using_pt=False):
    if using_pt:
        min_val = torch.min(arr)
        max_val = torch.max(arr)
    else:
        min_val = np.min(arr)
        max_val = np.max(arr)
    arr = (b - a) * (arr - min_val) / (max_val - min_val) + a
    return arr


def return_heat_map(scene_heatmap2):
    scene_heatmap2 = normalize(1, 0, scene_heatmap2) * 255
    scene_heatmap2 = cv2.applyColorMap(
        scene_heatmap2.astype(np.uint8), cv2.COLORMAP_JET
    )
    return scene_heatmap2


def normalize_shape(tensor_in):
    """Bring tensor from shape BxCxHxW to NxC"""
    return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)


def read_image_by_hloc(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image_by_hloc(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _strtobool(x):
    return bool(strtobool(x))


def get_patch(image, center, patch_size, padding=False):
    """
    Function to extract a patch from an image given the center coordinate and patch size.
    If padding is set to True, it adds padding to ensure the patch stays within the image boundaries.

    Parameters:
        image: numpy array representing the image.
        center: tuple containing the (x, y) coordinates of the center of the patch.
        patch_size: tuple containing the width and height of the patch.
        padding: boolean indicating whether to add padding to the patch to ensure it stays within the image boundaries.

    Returns:
        patch: numpy array representing the extracted patch.
    """
    x, y = center
    w, h = patch_size

    if padding:
        # Calculate padding to ensure the patch stays within the image boundaries
        top_pad = max(0, -y + h // 2)
        bottom_pad = max(0, y + h // 2 - image.shape[0])
        left_pad = max(0, -x + w // 2)
        right_pad = max(0, x + w // 2 - image.shape[1])

        # Pad the image
        image = cv2.copyMakeBorder(
            image,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        # Adjust the center coordinates
        x += left_pad
        y += top_pad

    # Calculate the top-left corner of the patch
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    # Calculate the bottom-right corner of the patch
    x2 = x1 + w
    y2 = y1 + h

    # Extract the patch from the image
    patch = image[y1:y2, x1:x2]

    return patch


def read_and_preprocess(name, conf):
    image = read_image_by_hloc(name, conf.grayscale)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]

    if conf.resize_max and (conf.resize_force or max(size) > conf.resize_max):
        scale = conf.resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image_by_hloc(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0
    return image, scale


def project_using_pose(gt_pose_inv_B44, intrinsics_B33, xyz):
    xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyzt = torch.from_numpy(xyzt).permute([1, 0]).float().cuda()

    gt_inv_pose_34 = gt_pose_inv_B44[0, :3]
    cam_coords = torch.mm(gt_inv_pose_34, xyzt)
    uv = torch.mm(intrinsics_B33[0], cam_coords)
    uv[2].clamp_(min=0.1)  # avoid division by zero
    uv = uv[0:2] / uv[2]
    uv = uv.permute([1, 0]).cpu().numpy()
    return uv


def project_using_pose2(example, xyzt):
    gt_inv_pose_34 = example[4][:3]
    cam_coords = torch.mm(gt_inv_pose_34, xyzt)
    uv = torch.mm(example[6], cam_coords)
    uv[2].clamp_(min=0.1)  # avoid division by zero
    uv = uv[0:2] / uv[2]
    uv = uv.permute([1, 0]).cpu().numpy()
    oob_mask = find_oob(example, uv)
    if np.sum(oob_mask) == 0:
        return None, oob_mask
    return uv[oob_mask], oob_mask
