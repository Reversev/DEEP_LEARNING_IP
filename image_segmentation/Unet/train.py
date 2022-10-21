import os
import time
import datetime

import torch

from src.unet import UNet
from utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
import utils.transforms as T
from utils.my_dataset import DriveDataset


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               **loader_args)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             collate_fn=val_dataset.collate_fn,
                                             **loader_args)

    model = create_model(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_iou = 0.
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes, epochs=args.epochs,
                                        lr_scheduler=lr_scheduler, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        mean_iou = float(val_info.split("\n")[-1].split(" ")[-1])
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"mean_iou: {mean_iou:.3f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if best_iou < mean_iou:
            best_iou = mean_iou
            best_epoch = epoch
            torch.save(save_file, args.checkpoint + "best_model.pth")
        else:
            torch.save(save_file, args.checkpoint + "model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}, best_iou: {} in epoch {}.".format(total_time_str, str(best_iou), str(best_epoch)))


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--data-path", default="../datasets/", help="dataset root")
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--batch-size", default=4, type=int)

    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=5e-4, type=float, help="weight decay (default: 1e-4)")

    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument("--checkpoint", default="./checkpoints/", help="dataset root")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    main(args)
