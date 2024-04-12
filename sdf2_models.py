import sys
sys.path.append("../sfd2")
from nets.sfd2 import ResSegNet, ResSegNetV2
from nets.extractor import extract_resnet_return
import torch
import os
import os.path as osp

confs = {
    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n3000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n3000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 3000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n2000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n2000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 2000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n1000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n1000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 1000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1024': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1024',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': False,
    },
}


def get_model(model_name, weight_path, use_stability=False):
    if model_name == 'ressegnet':
        model = ResSegNet(outdim=128, require_stability=use_stability).eval()
        model.load_state_dict(torch.load(weight_path)['model'], strict=True)
        extractor = extract_resnet_return
    if model_name == 'ressegnetv2':
        model = ResSegNetV2(outdim=128, require_stability=use_stability).eval()
        model.load_state_dict(torch.load(weight_path)['model'], strict=False)
        extractor = extract_resnet_return

    return model, extractor


def return_models():
    conf = confs["ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600"]
    model, extractor = get_model(model_name=conf['model']['name'],
                                 weight_path="../sfd2/weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth",
                                 use_stability=conf["model"]['use_stability'])
    model = model.cuda()
    return model, extractor, conf
    print("model: ", model)

    loader = ImageDataset(image_dir, conf['preprocessing'],
                          image_list=args.image_list,
                          mask_root=None)
    loader = torch.utils.data.DataLoader(loader, num_workers=4)

    feature_path = Path(export_dir, conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    with tqdm(total=len(loader)) as t:
        for idx, data in enumerate(loader):
            t.update()
            if tag is not None:
                if data['name'][0].find(tag) < 0:
                    continue
            pred = extractor(model, img=data["image"],
                             topK=conf["model"]["max_keypoints"],
                             mask=None,
                             conf_th=conf["model"]["conf_th"],
                             scales=conf["model"]["scales"],
                             )

            # pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            pred['descriptors'] = pred['descriptors'].transpose()

            t.set_postfix(npoints=pred['keypoints'].shape[0])
            # print(pred['keypoints'].shape)

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            # pred['descriptors'] = pred['descriptors'].T
            if 'keypoints' in pred.keys():
                size = np.array(data['image'].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            grp = feature_file.create_group(data['name'][0])
            for k, v in pred.items():
                # print(k, v.shape)
                grp.create_dataset(k, data=v)

            del pred

    feature_file.close()
    logging.info('Finished exporting features.')

    return feature_path