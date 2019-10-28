import argparse
import logging
from pathlib import Path

import tqdm

import torch as T


def init_dataloader(path):
    if isinstance(path, str):
        path = Path(path)

    augmentations = build_augmentations(num_classes=20, 256, 256)

    val_aug = augmentations['val']

    val_dataset = BDDSemSegDataset(
        path=path, split='val', num_classes=config['num_classes'],
        transform=val_aug
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16, pin_memory=True
    )


def main(args):
    data_loader = init_dataloader(args.dataset)

    num_classes = 20
    
    model = Model(num_classes)

    model.load_state_dict(T.load(args.checkpoint)['model_state_dict'])

    model.to('cuda')

    model.eval()

    with T.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(data_loader)):
            data = batch['img'].to('cuda')

            predict = model(data)['out']

            masks = T.argmax(predict, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset', type=str, help='Must be set of images')
    parser.add_argument('--output', type=str, default='./output')

    args = parser.parse_args()

    main(args)
