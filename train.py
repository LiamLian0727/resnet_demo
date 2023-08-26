import os
import time
import timm
import torch
import random
import logging
import argparse
import datetime
import importlib
import numpy as np

from thop import profile
from torch import optim as optim, nn
from timm.utils import accuracy, AverageMeter
from timm.scheduler import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy

from mmcv.cnn import get_model_complexity_info


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def set_logging(paths, time_str):
    logger = logging.getLogger(name='trainLogger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s %(name)s %(levelname)s] : %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(os.path.join(paths, f'train_{time_str}.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def custom_parse():
    parser = argparse.ArgumentParser(description='RepVGG')
    parser.add_argument('-m', '--module_name', type=str, required=True, help='Module Config')
    parser.add_argument('-t', '--train_config',
                        default="config.TrainConfig", type=str, required=False, help='Train Config')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='out_dir')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='random seed')
    args, unparsed = parser.parse_known_args()
    return args


def build_optimizer(lr=0.1, momentum=0.9, weight_decay=1e-4):
    return optim.SGD(lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)


def build_scheduler(optimizer, n_iter_per_epoch, num_epoch=120, warmup_epoch=5):
    return CosineLRScheduler(
        optimizer, lr_min=0, t_in_epochs=False,
        t_initial=int(num_epoch * n_iter_per_epoch),
        warmup_t=int(warmup_epoch * n_iter_per_epoch),
    )


@torch.no_grad()
def validate(epoch, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        if type(output) is dict:
            loss = criterion(output["out"], target) + 0.15 * criterion(output["aux1"], target) + \
                   0.33 * criterion(output["aux2"], target) + 0.66 * criterion(output["aux1"], target)
            output = output["out"]
        else:
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    logger.info(
        f'[epoch {epoch + 1}] Acc@1: {acc1_meter.avg:.2f}%, Acc@5: {acc5_meter.avg:.2f}%, loss: {loss_meter.avg:.4f}')
    return acc1_meter.avg, acc5_meter.avg


@torch.no_grad()
def test(data_loader, model):
    model.eval()
    test_len = len(data_loader)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start_time = None
    for idx, (images, target) in enumerate(data_loader):
        if id >= test_len // 2 and start_time == None:
            start = time.time()
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)
        if type(output) is dict:
            output = output["out"]
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    end = time.time()
    datetime.timedelta(seconds=int(end - start))
    logger.info(f'Acc@1: {acc1_meter.avg:.2f}%, Acc@5: {acc5_meter.avg:.2f}%, ' +
                f'Speed: {(end - start) / test_len - test_len // 2}')


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets_mixup = mixup_fn(samples, targets)
        outputs = model(samples)
        loss = criterion(outputs, targets_mixup)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), targets.size(0))
        acc5_meter.update(acc5.item(), targets.size(0))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        if idx % (num_steps // 5) == 0 or idx == num_steps - 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'epoch: {epoch + 1}/{config["epoch"]}, batch: {idx + 1}/{num_steps}, lr: {lr:.4f}, ' +
                f'loss {loss.item():.4f}, grad_norm {grad_norm:.2f}, Acc@1: {acc1_meter.avg:.2f}%, Acc@5: {acc5_meter.avg:.2f}%')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train(model, logger, train_iter, test_iter, optimizer, lr_scheduler, criterion, train_config, mixup_fn):
    logger.info("Start training")
    start_time = time.time()
    best_acc1 = 0
    for epoch in range(train_config["epoch"]):
        train_one_epoch(train_config, model, criterion, train_iter, optimizer, epoch, mixup_fn, lr_scheduler)
        acc1, acc5 = validate(epoch, test_iter, model)
        if epoch % (train_config["epoch"] // 5) == 0 or epoch >= int(train_config["epoch"] * 0.9):
            if acc1 >= best_acc1:
                torch.save(
                    {'model': model.state_dict()},
                    os.path.join(args.out_dir, f"{args.module_config.split('.')[-1]}_{time_str}_epoch{epoch + 1}.pth")
                )
                best_acc1 = max(best_acc1, acc1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = custom_parse()
    os.makedirs(args.out_dir, exist_ok=False)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    logger = set_logging(args.out_dir, time_str)
    setup_seed(args.seed)
    logger.info(f"Random Seed: {args.seed}")
    devices = try_all_gpus()
    logger.info(f"device in: {devices}")
    train_config = importlib.import_module(args.train_config).train_config
    logger.info(f"train config: \n{train_config}")
    module_name = args.module_name
    logger.info(f"module name: \n{module_name}")

    logger.info(f"Lording Dataset: {train_config['DATASET']}")
    if train_config["DATASET"] == "cifar100":
        from data.cifar100 import build_loader

        dataset_train, dataset_val, data_loader_train, data_loader_val, num_classes, mixup_fn = \
            build_loader(train_config["BATCH_SIZE"], train_config["NUM_WORKERS"], train_config["mixup_args"])
    logger.info(f"Lording Dataset {train_config['DATASET']} successfully")

    if module_name == "resnet_18":
        from resnet import resnet_18

        model = resnet_18(input_channels=3, num_classes=num_classes)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),))
    logger.info(f"params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}B (in Tensor(1, 3, 224, 224))")
    model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    logger.info(f"module structure : \n{model}")
    optimizer = build_optimizer(
        model, logger,
        lr=train_config['lr'], momentum=train_config['momentum'], weight_decay=train_config["weight_decay"]
    )
    lr_scheduler = build_scheduler(optimizer, len(data_loader_train), train_config["epoch"], train_config["warmup"])
    criterion = SoftTargetCrossEntropy()

    train(
        model, logger, data_loader_train, data_loader_val, optimizer, lr_scheduler, criterion, train_config, mixup_fn)
    torch.save(
        {'model': model.state_dict()},
        os.path.join(args.out_dir, f"{args.module_config.split('.')[-1]}_{time_str}_last.pth")
    )
