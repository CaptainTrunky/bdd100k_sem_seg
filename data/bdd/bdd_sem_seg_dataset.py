import logging
from pathlib import Path
import tqdm

from albumentations import (Compose, Resize, HorizontalFlip,
        RandomCrop, ToFloat, Normalize, OneOf, RandomScale, ShiftScaleRotate,
        Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
        IAAPiecewiseAffine, CLAHE, IAASharpen, IAAEmboss,
        RandomBrightnessContrast)

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


def build_augmentations(width, height, resize_strategy):
    basic_block = build_basic_transform()

    if resize_strategy == 'crop':
        resize_strategy = RandomCrop(width=width, height=height)
    elif resize_strategy == 'resize':
        resize_strategy = Resize(width=width, height=height, interpolation=cv2.INTER_CUBIC)
    else:
        raise RuntimeError(f'Unknown resize strategy {resize_strategy}')

    strong_aug = Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.0),
            ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
            ], p=0.3),
        HueSaturationValue(p=0.3),
        ])

    train = Compose([
        HorizontalFlip(),
        strong_aug,
        resize_strategy,
        basic_block
    ])

    val = Compose([
        resize_strategy,
        build_basic_transform()
    ])

    test = basic_block

    return {'train': train, 'val': val, 'test': test}


def init_dataloaders(config):
    augmentations = build_augmentations(config.width, config.height, config.resize_strategy)

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

        img_path = self.images[idx]

        if not img_path.exists(): 
            raise RuntimeError(f'failed to read {img_path.as_posix()}')

        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self.labels[idx]

        if not label_path.exists():
            raise RuntimeError(f'failed to read {label_path.as_posix()}')

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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)

            img = transformed['image']

        return {'img': img, 'idx': idx}

    def __len__(self):
        return len(self.images)
