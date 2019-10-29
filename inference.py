import argparse
import logging
from pathlib import Path

import cv2

import tqdm

import torch as T
from torch.utils.data import DataLoader

from models.bdd.SemSeg import SemSeg as Model

from data.bdd.bdd_sem_seg_dataset import build_augmentations, BDDSemSegTestDataset

from utils.vis import get_colormap


def init_dataloader(path):
    if isinstance(path, str):
        path = Path(path)

    augmentations = build_augmentations(width=256, height=256)

    test_aug = augmentations['test']

    test_dataset = BDDSemSegTestDataset(
        dataset_path=path, transform=test_aug
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16, pin_memory=True
    )

    return test_loader


def main(args):
    data_loader = init_dataloader(args.dataset)

    num_classes = 20
    
    model = Model(num_classes)

    model.load_state_dict(T.load(args.checkpoint)['model_state_dict'])

    device = 'cpu'

    model.to(device)

    model.eval()

    colormap = get_colormap()

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
                img = colormap[imgs[i, :, :]]

                p = data_loader.dataset.images[indices[i]]

                cv2.imwrite((output / f'{p.name}').as_posix(), img, [cv2.IMREAD_UNCHANGED])

            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset', type=str, help='Must be set of images')
    parser.add_argument('--output', type=str, default='./output')

    args = parser.parse_args()

    main(args)
