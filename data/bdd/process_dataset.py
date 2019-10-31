import argparse
import logging

from pathlib import Path

import cv2
import numpy as np
from  albumentations import Resize
import tqdm


logging.basicConfig(level=logging.INFO)


def main(args):
    root = args.dataset_root

    if isinstance(root, str):
        root = Path(root)

    if not root.exists() or not root.is_dir():
        raise RuntimeError(f'{root.as_posix()} is not a valid path')

    output = args.output

    if isinstance(output, str):
        output = Path(output)

    if output.exists():
        raise RuntimeError(f'{output.as_posix()} already exists')

    output.mkdir()

    images = root / 'images'

    if not images.exists() or not images.is_dir():
        raise RuntimeError(f'{images.as_posix()} is not a valid path')

    labels = root / 'labels'

    if not labels.exists() or not labels.is_dir():
        raise RuntimeError(f'{labels.as_posix()} is not a valid path')

    output_images = output / 'images'

    output_images.mkdir()
    
    output_labels = output / 'labels'

    output_labels.mkdir()

    for split in images.iterdir():
        logging.info(f'cur split: {split.stem}')

        if split.is_file():
            continue

        if split.stem == 'test':
            continue

        output_images_split = output_images / split.stem

        output_images_split.mkdir()

        output_labels_split = output_labels / split.stem

        output_labels_split.mkdir()

        for img in tqdm.tqdm(list(split.iterdir())):
            if not img.name.endswith('jpg'):
                continue

            img_id = img.stem

            label_id = img_id + '_train_id.png'
 
            label = cv2.imread((labels / split.stem / label_id).as_posix(), cv2.IMREAD_UNCHANGED)
  
            output_label = np.full(shape=label.shape, fill_value=255, dtype=np.uint8)
            for idx, valid_id in enumerate(args.valid_ids):
                mask = (label == valid_id).astype(bool)

                if np.any(mask):
                    output_label[mask] = idx

                if valid_id in args.merge_ids:
                    output_label[mask] = args.merge_ids[0]

            (output_images_split / img.name).write_bytes(img.read_bytes())
            cv2.imwrite((output_labels_split / label_id).as_posix(), output_label, [cv2.IMREAD_UNCHANGED])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_root', type=str)

    parser.add_argument('valid_ids', type=int, nargs='+')
    parser.add_argument('--merge_ids', type=int, nargs='+')

    parser.add_argument('--output', type=str, default='./processed')

    args = parser.parse_args()

    main(args)
