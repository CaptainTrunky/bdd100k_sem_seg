import logging
from pathlib import Path

import cv2
import numpy as np

import random

import tqdm

import torch as T
import torch.nn.functional as F
import torch.optim as optim

from data.bdd.bdd_sem_seg_dataset import init_dataloaders

from metrics.generic import get_all_metrics

from models.bdd.SemSeg import SemSeg as Model

from thirdparty.ranger.ranger import Ranger
import thirdparty.LovaszSoftmax.pytorch.lovasz_losses as L

from utils.vis import get_colormap

from sacred import Experiment
from sacred.observers import MongoObserver

from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO)

ex = Experiment('bdd_sem_seg')

ex.observers.append(MongoObserver.create())

# ignore, moving car, parked car, person, semaphore, road
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


def train(config, loaders):
    train_loader = loaders['train']
    val_loader = loaders['val']

    logging.info(f'train: {len(train_loader)}, val: {len(val_loader)}')

    model = Model(config.num_classes)

    optimizer = Ranger(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(config.device)

    criterion = None

    if config.loss == 'wce':
        weights = T.ones((config.num_classes,)).float()

        weights[1] = 5.0
        weights[6] = 10.0
        weights[7] = 10.0
        weights[11] = 10.0
        weights[14] = 5.0
        weights[15] = 5.0

        criterion = T.nn.CrossEntropyLoss(weight=weights.to(config.device), ignore_index=255).to(config.device)
    elif config.loss == 'lovasz_softmax':
        criterion = lambda input, target: L.lovasz_softmax(probas=input, labels=target, ignore=255)
    elif config.loss == 'mixed':
        weights = T.ones((config.num_classes,)).float()

        weights[1] = 5.0
        weights[6] = 10.0
        weights[7] = 10.0
        weights[11] = 10.0
        weights[14] = 5.0
        weights[15] = 5.0

        wce = T.nn.CrossEntropyLoss(weight=weights.to(config.device), ignore_index=255).to(config.device)
   
        lovasz = lambda input, target: L.lovasz_softmax(probas=input, labels=target, ignore=255)

        criterion = lambda input, target: 1.25 * lovasz(input, target) + wce(input, target)
    else:
        raise RuntimeError(f'Unknown loss function: {config.loss}')

    run = config.manager.run

    best_score = 0

    weights_path = config.manager.weights

    if not weights_path.exists():
        weights_path.mkdir()

    main_bar = tqdm.tqdm(range(1, config.epochs + 1), desc='epoch')
    for epoch in main_bar:
        train_loss = train_epoch(model, optimizer, criterion, train_loader, epoch, config.manager)
        val_loss = val_step(model, val_loader, criterion, epoch, config.manager)

        scheduler.step() 

        run.log_scalar('train.lr', scheduler.optimizer.param_groups[0]['lr'])

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


def train_epoch(model, optimizer, criterion, train_loader, epoch, manager):
    model.train()

    losses = []
    acc = 0
    loss = None

    device = next(model.parameters()).device

    for idx, batch in enumerate(tqdm.tqdm(train_loader, desc='train')):
        data = batch['img'].to(device)
        labels = batch['label'].long().to(device)

        predict = model(data)['out']

        loss = criterion(input=predict, target=labels)

        with T.no_grad():
            losses.append(loss.mean().item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        manager.run.log_scalar('train.cross_entropy', np.mean(losses))

    return np.mean(losses)


def val_step(model, val_loader, criterion, epoch, manager):
    model.eval()

    losses = []

    device = next(model.parameters()).device

    metrics = get_all_metrics()
    metrics_values = dict()

    with T.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(val_loader, desc='val')):
            data = batch['img'].to(device)
            labels = batch['label'].long().to(device)

            predict = model(data)['out']

            loss = criterion(input=predict, target=labels)

            losses.append(loss.mean().item())

            masks = T.argmax(predict, dim=1)
    
            for label_id, label_name in VALID_MASK_IDS.items():
                for name, metric in metrics.items():
                    val = metric.compute(
                        np.expand_dims(masks.cpu().numpy(), 1),
                        np.expand_dims(labels.cpu().numpy(), 1),
                        label_id
                    )

                    metric_name = f'{label_name}_{name}'

                    if metric_name not in metrics_values:
                        metrics_values[metric_name] = [val]
                    else:
                        metrics_values[metric_name].append(val)

    manager.run.log_scalar('val.cross_entropy', np.mean(losses))

    ious = []
    for metric_name, vals in metrics_values.items():
        manager.run.log_scalar(f'test.{metric_name}', np.mean(vals))
  
        if '_iou' in metric_name:
            m = np.mean(vals)
            logging.info(f'test.{metric_name}: {m}')

            ious.append(m)

    mean_ious = np.mean(ious)
    logging.info(f'test.mean_iou: {mean_ious}')
    manager.run.log_scalar('test.mean_iou', mean_ious)

    if True:
        imgs = masks.detach().cpu().numpy()

        colormap = get_colormap(20)
        # dump colored images
        for i in range(3):
            img = imgs[i, :, :]
                 
            rgb = (np.moveaxis(batch['img'][i, ...].numpy(), 0, 2) * 255.0).astype(np.uint8)
            label = batch['label'][i, ...].numpy()
            label = np.repeat(np.expand_dims(label, 2), 3, axis=2)
                
            colored_masks = np.hstack((rgb, label, colormap[img]))

            cv2.imwrite(
                (manager.artifacts / f'{epoch}_{i}.png').as_posix(),
                colored_masks, [cv2.IMREAD_UNCHANGED]
            )

    return np.mean(losses)


@ex.automain
def ex_main(_run, trainer_config):
    random.seed(17)
    np.random.seed(17)

    T.manual_seed(17)

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    from munch import Munch
    trainer_config = Munch(trainer_config)

    from core.manager import ExperimentManager
    trainer_config.manager = ExperimentManager(trainer_config, _run)

    loaders = init_dataloaders(config=trainer_config)

    train(trainer_config, loaders)
