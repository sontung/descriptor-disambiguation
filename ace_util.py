import random
from distutils.util import strtobool

import PIL
import cv2
import numpy as np
import torch
from tqdm import tqdm


def read_nvm_file(file_name):
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
    rgb_arr = np.zeros((nb_points, 3), np.float64)
    # nb_points = 100
    for j in tqdm(range(nb_points), desc="Reading points"):
        point_info = lines[5 + nb_cameras + j].split(" ")
        x, y, z, r, g, b, nb_features = point_info[:7]
        x, y, z = map(float, [x, y, z])
        xyz_arr[j] = [x, y, z]
        rgb_arr[j] = [r, g, b]
        features_info = point_info[7:]
        nb_features = int(nb_features)
        for k in range(nb_features):
            image_id, _, u, v = features_info[k * 4 : (k + 1) * 4]
            image_id = int(image_id)
            u, v = map(float, [u, v])
            image2points.setdefault(image_id, []).append(j)
            image2uvs.setdefault(image_id, []).append([u, v])

    return xyz_arr, image2points, image2name, image2pose, image2info, image2uvs, rgb_arr


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
    scale = 1
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

    gt_inv_pose_34 = gt_pose_inv_B44[0, :3].float()
    cam_coords = torch.mm(gt_inv_pose_34.float(), xyzt)
    uv = torch.mm(intrinsics_B33[0].float(), cam_coords)
    uv[2].clamp_(min=0.1)  # avoid division by zero
    uv = uv[0:2] / uv[2]
    uv = uv.permute([1, 0]).cpu().numpy()
    return uv
