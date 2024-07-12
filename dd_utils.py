import math

import cv2
import faiss
import numpy as np
import skimage
from scipy.spatial.transform import Rotation as Rotation
from skimage.transform import rotate
from types import SimpleNamespace
from kornia.feature import DeDoDe
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load


def cluster_by_faiss_kmeans(x, nb_clusters, verbose=False):
    niter = 20
    d = x.shape[1]

    kmeans = faiss.Kmeans(d, int(nb_clusters), niter=niter, verbose=verbose)
    kmeans.train(x)

    _, indices = kmeans.index.search(x, 1)

    indices = indices.flatten()
    return indices, kmeans.index


def return_pose_mat(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    # convert world->cam to cam->world for evaluation
    pose_4x4_inv = np.linalg.inv(pose_4x4)
    return pose_4x4_inv


def return_pose_mat_no_inv(pose_q, pose_t):
    pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
    pose_R = Rotation.from_quat(pose_q).as_matrix()

    pose_4x4 = np.identity(4)
    pose_4x4[0:3, 0:3] = pose_R
    pose_4x4[0:3, 3] = pose_t

    return pose_4x4


def return_pixel_grid_dsac():
    pixel_grid = torch.zeros((2, math.ceil(5000 / 8), math.ceil(5000 / 8)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * 8 + 4
            pixel_grid[1, y, x] = y * 8 + 4
    return pixel_grid


def rotate_image(t, angle_, order, mode="constant"):
    t = t.permute(1, 2, 0).numpy()
    t = rotate(t, angle_, order=order, mode=mode)
    t = torch.from_numpy(t).permute(2, 0, 1).float()
    return t


def transform_kp(kp, max_size, image_ori, image_resize, angle):
    height = image_ori.shape[1]
    width = image_ori.shape[0]
    scale = max_size / min([height, width])
    kp = kp * scale

    # if angle == 0 and type(angle) == int:
    #     return kp.astype(np.int32)

    h = image_resize.size(1)
    w = image_resize.size(2)

    translate = {"x": 0, "y": 0}

    shear = {"x": -0.0, "y": -0.0}
    scale = {"x": 1.0, "y": 1.0}

    rotate = -angle
    shift_x = w / 2 - 0.5
    shift_y = h / 2 - 0.5

    matrix_to_topleft = skimage.transform.SimilarityTransform(
        translation=[-shift_x, -shift_y]
    )
    matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
    matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
    matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
    matrix_transforms = skimage.transform.AffineTransform(
        scale=(scale["x"], scale["y"]),
        translation=(translate["x"], translate["y"]),
        rotation=np.deg2rad(rotate),
        shear=np.deg2rad(shear["x"]),
    )
    matrix_to_center = skimage.transform.SimilarityTransform(
        translation=[shift_x, shift_y]
    )
    matrix = (
        matrix_to_topleft
        + matrix_shear_y_rot
        + matrix_shear_y
        + matrix_shear_y_rot_inv
        + matrix_transforms
        + matrix_to_center
    )

    kp2 = np.copy(kp)
    kp2 = np.expand_dims(kp2, 0)
    kp2 = cv2.transform(kp2, matrix.params[:2]).squeeze()

    return kp2.astype(np.int32)


def transform_kp_aug_fast(
    kp_indices, image_height, scale_factor, image, image_transformed, angle
):
    keypoints = transform_kp(
        kp_indices,
        int(image_height * scale_factor),
        image,
        image_transformed,
        angle,
    )

    keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
    kp_map = np.zeros(
        [image_transformed.shape[1], image_transformed.shape[2]], dtype=np.int8
    )
    mask1 = np.bitwise_and(
        0 <= keypoints[:, 0], keypoints[:, 0] < image_transformed.shape[1]
    )
    mask2 = np.bitwise_and(
        0 <= keypoints[:, 1], keypoints[:, 1] < image_transformed.shape[2]
    )
    mask = np.bitwise_and(mask1, mask2)
    valid_keypoints = keypoints[mask]
    kp_map[valid_keypoints[:, 0], valid_keypoints[:, 1]] = 1
    return keypoints, valid_keypoints, kp_map, mask


def hloc_conf_for_all_models():
    conf = {
        "superpoint": {
            "output": "feats-superpoint-n4096-r1024",
            "model": {
                "name": "superpoint",
                "nms_radius": 3,
                "max_keypoints": 4096,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1024,
            },
        },
        "r2d2": {
            "output": "feats-r2d2-n5000-r1024",
            "model": {
                "name": "r2d2",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1024,
            },
        },
        "d2net": {
            "output": "feats-d2net-ss",
            "model": {
                "name": "d2net",
                "multiscale": False,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "sift": {
            "output": "feats-sift",
            "model": {"name": "dog"},
            "preprocessing": {
                "grayscale": True,
                "resize_max": 1600,
            },
        },
        "disk": {
            "output": "feats-disk",
            "model": {
                "name": "disk",
                "max_keypoints": 5000,
            },
            "preprocessing": {
                "grayscale": False,
                "resize_max": 1600,
            },
        },
        "netvlad": {
            "output": "global-feats-netvlad",
            "model": {"name": "netvlad"},
            "preprocessing": {"resize_max": 1024},
        },
        "openibl": {
            "output": "global-feats-openibl",
            "model": {"name": "openibl"},
            "preprocessing": {"resize_max": 1024},
        },
        "eigenplaces": {
            "output": "global-feats-eigenplaces",
            "model": {"name": "eigenplaces"},
            "preprocessing": {"resize_max": 1024},
        },
    }

    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }
    return conf, default_conf


def read_kp_and_desc(name, features_h5):
    img_id = "/".join(name.split("/")[-2:])
    try:
        grp = features_h5[img_id]
    except KeyError:
        grp = features_h5[name]

    pred = {k: np.array(v) for k, v in grp.items()}
    scale = pred["scale"]
    keypoints = (pred["keypoints"] + 0.5) / scale - 0.5
    if "descriptors" in pred:
        descriptors = pred["descriptors"].T
    else:
        descriptors = None
    return keypoints, descriptors


def read_desc_only(name, features_h5):
    img_id = "/".join(name.split("/")[-2:])
    try:
        grp = features_h5[img_id]
    except KeyError:
        grp = features_h5[name]

    if "descriptors" in grp:
        descriptors = np.array(grp["descriptors"]).T
    else:
        descriptors = None
    return descriptors


def read_global_desc(name, global_features_h5):
    img_id = "/".join(name.split("/")[-2:])
    try:
        desc = np.array(global_features_h5[name]["global_descriptor"])
    except KeyError:
        desc = np.array(global_features_h5[img_id]["global_descriptor"])
    return desc


def write_to_h5_file(fd, name, dict_):
    img_id = "/".join(name.split("/")[-2:])
    name = img_id
    try:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for k, v in dict_.items():
            grp.create_dataset(k, data=v)
    except OSError as error:
        if "No space left on device" in error.args[0]:
            print("No space left")
            del grp, fd[name]
        raise error


def prepare_encoders(local_desc_model, retrieval_model, global_desc_dim):
    conf, default_conf = hloc_conf_for_all_models()

    try:
        model_dict = conf[local_desc_model]["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        encoder = Model(model_dict).eval().to(device)
        conf_ns = SimpleNamespace(**{**default_conf, **conf})
        conf_ns.grayscale = conf[local_desc_model]["preprocessing"]["grayscale"]
        conf_ns.resize_max = conf[local_desc_model]["preprocessing"]["resize_max"]
    except KeyError:
        if local_desc_model == "sfd2":
            conf_ns = SimpleNamespace(**{**default_conf, **conf})
            conf_ns.grayscale = False
            conf_ns.resize_max = 1600
            import sfd2_models

            encoder = sfd2_models.return_models()
        elif local_desc_model == "dedode":
            encoder = DeDoDe.from_pretrained(
                detector_weights="L-upright", descriptor_weights="B-upright"
            )
            conf_ns = SimpleNamespace(**{**default_conf, **conf})
            encoder.cuda()
        elif local_desc_model == "how":
            from how_model import HowModel
            conf_ns = SimpleNamespace(**{**default_conf, **conf})
            encoder = HowModel()
        elif local_desc_model == "xfeat":
            from xfeat_model import XfeatModel
            conf_ns = SimpleNamespace(**{**default_conf, **conf})
            encoder = XfeatModel()
        else:
            raise NotImplementedError

    if retrieval_model == "mixvpr":
        from mix_vpr_model import MVModel

        encoder_global = MVModel(global_desc_dim)
        conf_ns_retrieval = None
    elif retrieval_model == "crica":
        from crica_model import CricaModel

        encoder_global = CricaModel()
        conf_ns_retrieval = None
    elif retrieval_model == "salad":
        from salad_model import SaladModel

        encoder_global = SaladModel()
        conf_ns_retrieval = None
    elif retrieval_model == "gcl":
        from gcl_model import GCLModel

        encoder_global = GCLModel()
        conf_ns_retrieval = None
    elif retrieval_model == "dino":
        from dino_model import DinoModel

        encoder_global = DinoModel()
        conf_ns_retrieval = None
    else:
        model_dict = conf[retrieval_model]["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, model_dict["name"])
        if retrieval_model == "eigenplaces":
            model_dict.update(
                {
                    "variant": "EigenPlaces",
                    "backbone": "ResNet101",
                    "fc_output_dim": global_desc_dim,
                }
            )
            encoder_global = Model(model_dict).eval().to(device)
            encoder_global.conf["name"] = f"eigenplaces_{model_dict['backbone']}"
        else:
            encoder_global = Model(model_dict).eval().to(device)
        conf_ns_retrieval = SimpleNamespace(**{**default_conf, **conf})
        conf_ns_retrieval.resize_max = conf[retrieval_model]["preprocessing"][
            "resize_max"
        ]
    return encoder, conf_ns, encoder_global, conf_ns_retrieval


def concat_images_different_sizes(images):
    # get maximum width
    ww = max([du.shape[0] for du in images])

    # pad images with transparency in width
    new_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        w1 = img.shape[0]
        img = cv2.copyMakeBorder(
            img, 0, ww - w1, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
        )
        new_images.append(img)

    # stack images vertically
    result = cv2.hconcat(new_images)
    return result
