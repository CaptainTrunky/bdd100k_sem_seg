import logging
from pathlib import Path

import cv2
import numpy as np

import random
import functools

from munch import Munch

import tqdm

import torch as T
import torch.nn.functional as F
import torch.optim as optim

from data.bdd.bdd_sem_seg_dataset import init_dataloaders

from models.bdd.SemSeg import SemSeg as Model

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO)

ex = Experiment('bdd_sem_seg')

ex.observers.append(MongoObserver.create())

# writer = SummaryWriter(logdir='/home/sbykov/workspace/ml/runs/bdd')
writer = None

@functools.lru_cache(maxsize=1)
def get_colormap(max_class_id=256):
    colormap = np.zeros((max_class_id, 3), dtype=np.uint8)

    colormap[0, :] = [255, 0, 0]
    colormap[6, :] = [128, 128, 0]
    colormap[11, :] = [0, 255, 0]
    colormap[13, :] = [0, 0, 255]
    colormap[14, :] = [0, 128, 128]
    colormap[255, :] = [255, 255, 255] 

    return colormap


# ignore, moving car, parked car, person, semaphore, road
VALID_MASK_IDS = {0, 6, 11, 13, 14, 255}

EARLY_STOP = 20000


@ex.config
def ex_config():
    root_path = Path('/home/s-bykov/workspace/datasets')

    trainer_config = {
        'dataset_path': root_path / 'bdd100k' / 'seg',
        'device': 'cuda',
        'num_classes': 20,
        'height': 256,
        'width': 256,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'shuffle': True
    }


def train(config, loaders):
    train_loader = loaders['train']
    val_loader = loaders['val']

    logging.info(f'train: {len(train_loader)}, val: {len(val_loader)}')

    model = Model(config.num_classes)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.to(config.device)

    criterion = T.nn.CrossEntropyLoss(ignore_index=255).to(config.device)

    logger = config.run

    best_score = 0

    weights_path = Path('./weights')

    if not weights_path.exists():
        weights_path.mkdir()

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25)

    main_bar = tqdm.tqdm(range(1, config.num_epochs + 1), desc='epoch')
    for epoch in main_bar:
        train_loss = train_epoch(model, optimizer, criterion, train_loader, epoch, logger)
        val_loss = val_step(model, val_loader, criterion, epoch, logger)

        if epoch % 5 == 0:
            model_name = f"./best_model_{epoch}_{str(val_loss).replace('.', 'd')}.pth"

            T.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                },
                weights_path / model_name
            )

        main_bar.set_postfix(val_loss=val_loss, train_loss=train_loss.item(), refresh=False)


def train_epoch(model, optimizer, criterion, train_loader, epoch, logger):
    model.train()

    losses = []
    acc = 0
    total_samples = 0
    loss = None

    device = next(model.parameters()).device

    for idx, batch in enumerate(tqdm.tqdm(train_loader, desc='train')):
        data = batch['img'].to(device)
        labels = batch['label'].long().to(device)

        predict = model(data)['out']

        loss = criterion(input=predict, target=labels)

        losses.append(loss.mean().item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with T.no_grad():
            acc += T.argmax(predict, dim=1).eq(labels).sum().item()
            total_samples += data.size(0)

        if idx > EARLY_STOP:
            break

    if writer:
        writer.add_scalar('train.accuracy', 100 * acc / total_samples, global_step=epoch)
        writer.add_scalar('train.cross_entropy', np.mean(losses), global_step=epoch)
    else:
        logger.log_scalar('train.accuracy', 100 * acc / total_samples)
        logger.log_scalar('train.cross_entropy', np.mean(losses))

    return np.mean(losses)


def val_step(model, val_loader, criterion, epoch, logger):
    model.eval()

    losses = []
    acc = 0
    total_samples = 0

    device = next(model.parameters()).device

    with T.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(val_loader, desc='val')):
            data = batch['img'].to(device)
            labels = batch['label'].long().to(device)

            predict = model(data)['out']

            loss = criterion(input=predict, target=labels)

            losses.append(loss.mean().item())

            masks = T.argmax(predict, dim=1)
            acc += masks.eq(labels).sum().item()
            total_samples += data.size(0)

            if idx > EARLY_STOP:
                break

    if writer:
        writer.add_scalar('val.accuracy', 100 * acc / total_samples, global_step=epoch)
    else:
        logger.log_scalar('val.cross_entropy', np.mean(losses))
        logger.log_scalar('val.accuracy', 100 * acc / total_samples)

        imgs = masks.detach().cpu().numpy()

        colormap = get_colormap()
        # dump colored images
        for i in range(3):
            img = imgs[i, :, :]

            rgb = (np.moveaxis(batch['img'][i, ...].numpy(), 0, 2) * 255.0).astype(np.uint8)
            colored_masks = np.hstack((rgb, colormap[img]))

            cv2.imwrite(f'./images/{epoch}_{i}.png', colored_masks, [cv2.IMREAD_UNCHANGED])

    return np.mean(losses)


@ex.automain
def ex_main(_run, trainer_config):
    random.seed(17)
    np.random.seed(17)

    T.manual_seed(17)

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    trainer_config['run'] = _run

    trainer_config = Munch(trainer_config)

    loaders = init_dataloaders(config=trainer_config)

    train(trainer_config, loaders)
