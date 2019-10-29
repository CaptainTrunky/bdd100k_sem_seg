import logging
from pathlib import Path
import tqdm

from albumentations import Compose, Resize, HorizontalFlip,\
        RandomCrop, ToFloat, Normalize

from albumentations.pytorch import ToTensor, ToTensorV2

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)


def build_basic_transform():
    normalization = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)}

    return Compose([
        Normalize(mean=normalization['mean'], std=normalization['std']),
        ToTensorV2()
    ])


def build_augmentations(width, height):
    basic_block = build_basic_transform()

    train = Compose([
        HorizontalFlip(),
        RandomCrop(width=width, height=height),
        basic_block
    ])

    val = Compose([
        RandomCrop(width=width, height=height),
        build_basic_transform()
    ])

    test = basic_block

    return {'train': train, 'val': val, 'test': test}


def init_dataloaders(config):
    augmentations = build_augmentations(config.width, config.height)

    dataset_path = config.dataset_path

    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    train_aug = augmentations['train']
    train_dataset = BDDSemSegDataset(
        path=dataset_path, split='train', num_classes=config['num_classes'],
        transform=train_aug
    )

    val_aug = augmentations['val']
    val_dataset = BDDSemSegDataset(
        path=dataset_path, split='val', num_classes=config['num_classes'],
        transform=val_aug
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=config.shuffle
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, pin_memory=True
    )

    return {'train': train_loader, 'val': val_loader}


class BDDSemSegDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()

        self.dataset_path = kwargs['path']

        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)

        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            raise RuntimeError(f'{self.dataset_path} is not valid')

        self.split = kwargs['split']

        self._load_samples()

        self._set_num_classes(kwargs)

        logging.info(f'{self.split}: got {len(self.images)} images, {self.num_classes} classes')

        self.transform = kwargs['transform']

    def _set_num_classes(self, kwargs):
        self.num_classes = -1

        if kwargs['num_classes']:
            self.num_classes = kwargs['num_classes']
        else:
            classes = set()
            for label in tqdm.tqdm(self.labels):
                p = self.dataset_path / 'labels' / self.split / label

                mask = cv2.imread(p.as_posix(), cv2.IMREAD_UNCHANGED)

                classes.update(
                    np.unique(mask, return_counts=False, return_index=False, return_inverse=False)
                )

            self.num_classes = len(classes)

    def _load_images(self):
        images = self.dataset_path / 'images' / self.split
        if not images.exists() or not images.is_dir():
            raise RuntimeError(f'{images.as_posix()} is not valid')

        self.images = [p for p in sorted(images.glob('*.jpg'))]

    def _load_labels(self):
        labels = self.dataset_path / 'labels' / self.split
        if not labels.exists() or not labels.is_dir():
            raise RuntimeError(f'{labels.as_posix()} is not valid')

        self.labels = [p for p in sorted(labels.glob('*.png'))]

    def _load_samples(self):
        self._load_images()
        self._load_labels()

        if len(self.images) != len(self.labels):
            msg = f'images num {len(self.images)} != '
            msg += f'!= labels num {len(self.labels)}'

            raise RuntimeError(msg)

    def __getitem__(self, idx):
        if self.images[idx].stem != self.labels[idx].stem.split('_')[0]:
            msg = f'Inconsistent image and label ids: '
            msg += f"{self.images[idx].stem}, {self.labels[idx].stem.split('_')[0]}"

            raise RuntimeError(msg)

        img_path = self.dataset_path / 'images' / self.split / self.images[idx]
        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(np.uint8)

        label_path = self.dataset_path / 'labels' / self.split / self.labels[idx]
        label = cv2.imread(label_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=img, mask=label)

            img = transformed['image']
            label = transformed['mask']

        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.images)


class BDDSemSegTestDataset(Dataset):
    def __init__(self, **kwargs):
        self.dataset_path = kwargs['dataset_path']
        self.transform = kwargs['transform']

        self._load_images()

    def _load_images(self):
        images = self.dataset_path

        if not images.exists() or not images.is_dir():
            raise RuntimeError(f'{images.as_posix()} is not valid')

        self.images = [p for p in sorted(images.glob('*.jpg'))]

        logging.info(f'Got {len(self.images)} images for testing')

    def __getitem__(self, idx):
        img_path = self.dataset_path / self.images[idx]
        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=img)

            img = transformed['image']

        return {'img': img, 'idx': idx}

    def __len__(self):
        return len(self.images)
