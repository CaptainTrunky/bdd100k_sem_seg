import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import tqdm

import torch as T
from torch.utils.data import DataLoader

from models.bdd.SemSeg import SemSeg as Model

from data.bdd.bdd_sem_seg_dataset import build_augmentations, BDDSemSegTestDataset

from utils.vis import get_colormap, blend_rgb_with_mask

VALID_MASK_IDS = {
    0: 'road',
    1: 'sidewalk',
    6: 'semaphore',
    7: 'sign',
    11: 'person',
    13: 'car',
    14: 'truck',
    15: 'bus'
}

VALID_MASK_IDS = {
    0: 'road',
    1: 'semaphore',
    2: 'sign',
    3: 'person',
    4: 'car',
    5: 'background',
}

def init_dataloader(path):
    if isinstance(path, str):
        path = Path(path)

    augmentations = build_augmentations(width=320, height=180, resize_strategy='resize')

    test_aug = augmentations['test']

    test_dataset = BDDSemSegTestDataset(
        dataset_path=path, transform=test_aug
    )

    test_loader = DataLoader(
        test_dataset, batch_size=8
    )

    return test_loader


def main(args):
    data_loader = init_dataloader(args.dataset)

    num_classes = 6

    device = 'cuda'

    model = Model(num_classes)

    if device == 'cpu':
        model.load_state_dict(T.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    else:
        model.load_state_dict(T.load(args.checkpoint)['model_state_dict'])

    model.to(device)

    model.eval()

    colormap = get_colormap(num_classes)

    print('colormap')

    for idx, color in enumerate(colormap):
        print(idx, color)

    output = args.output

    if isinstance(output, str):
        output = Path(output)

    if output.exists():
        raise RuntimeError(f'{output.as_posix()} already exists')

    output.mkdir()

    with T.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(data_loader)):
            data = batch['img'].to(device)

            predict = model(data)['out']

            masks = T.argmax(predict, dim=1)

            imgs = masks.detach().cpu().numpy()

            indices = batch['idx']

            for i in range(imgs.shape[0]):
                p = data_loader.dataset.images[indices[i]]
                
                rgb = cv2.imread(p.as_posix(), cv2.IMREAD_UNCHANGED)

                src_mask = imgs[i, :, :]

                mask_to_draw = np.zeros_like(src_mask)

                for valid_id in VALID_MASK_IDS.keys():
                    if valid_id == 5:
                        continue

                    mask_to_draw[src_mask == valid_id] = 1

                mask_to_draw = np.repeat(np.expand_dims(mask_to_draw, 2), 3, axis=2)

                img = colormap[src_mask] * mask_to_draw

                blended = blend_rgb_with_mask(rgb, img)
                cv2.imwrite((output / f'{p.name}').as_posix(), blended, [cv2.IMREAD_UNCHANGED])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset', type=str, help='Must be set of images')
    parser.add_argument('--output', type=str, default='./output')

    args = parser.parse_args()

    main(args)
