import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from modeling.sync_batchnorm.replicate import patch_replication_callback
from mypath import Path
from utils.calculate_weights import calculate_weigths_labels
from utils.checkpoint import load_checkpoint, state_dict_from_model
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary


JIJIE_DEFAULT_EPOCHS = 150
JIJIE_DEFAULT_LR = 0.007


def parse_selected_classes(raw_value):
    if raw_value in (None, ""):
        return None

    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        class_id = int(item)
        if class_id <= 0:
            raise ValueError("selected class ids must be positive integers, got {}".format(class_id))
        if class_id not in values:
            values.append(class_id)
    return values or None


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        loader_kwargs = {"num_workers": args.workers, "pin_memory": args.cuda}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **loader_kwargs)

        model = DeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
        )

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
        ]
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(
                Path.db_root_dir(args.dataset),
                args.dataset + "_classes_weights.npy",
            )
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = model
        self.optimizer = optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

            checkpoint = load_checkpoint(
                args.resume,
                self.model,
                optimizer=None if args.ft else self.optimizer,
                map_location="cpu",
            )
            args.start_epoch = checkpoint.get("epoch", 0)
            self.best_pred = checkpoint.get("best_pred", 0.0)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))

        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        vis_interval = max(1, num_img_tr // 10)

        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description("Train loss: {:.3f}".format(train_loss / (i + 1)))
            global_step = i + num_img_tr * epoch
            self.writer.add_scalar("train/total_loss_iter", loss.item(), global_step)

            if i % vis_interval == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print("[Epoch: {}, numImages: {}]".format(epoch, len(self.train_loader.dataset)))
        print("Loss: {:.3f}".format(train_loss))

        if self.args.no_val:
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": state_dict_from_model(self.model),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best=False,
            )

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Val loss: {:.3f}".format(test_loss / (i + 1)))

            pred = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target_np, pred)

        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)
        self.writer.add_scalar("val/Acc", acc, epoch)
        self.writer.add_scalar("val/Acc_class", acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", fwiou, epoch)

        is_best = miou > self.best_pred
        if is_best:
            self.best_pred = miou

        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict_from_model(self.model),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
            is_best=is_best,
        )

        print("Validation:")
        print("[Epoch: {}, numImages: {}]".format(epoch, len(self.val_loader.dataset)))
        print(
            "Acc:{}, Acc_class:{}, mIoU:{}, fwIoU:{}, best_mIoU:{}".format(
                acc,
                acc_class,
                miou,
                fwiou,
                self.best_pred,
            )
        )
        print("Loss: {:.3f}".format(test_loss))


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Training")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["resnet", "xception", "drn", "mobilenet"],
        help="backbone name",
    )
    parser.add_argument("--out-stride", type=int, default=16, help="network output stride")
    parser.add_argument(
        "--dataset",
        type=str,
        default="jijie",
        choices=[
            "pascal",
            "coco",
            "cityscapes",
            "her2_region",
            "feiai_region",
            "beiertongbxr_region",
            "qidai_region",
            "taimo_region",
            "prostate_tls",
            "jijie",
        ],
        help="dataset name",
    )
    parser.add_argument("--use-sbd", action="store_true", default=True, help="use SBD dataset for Pascal")
    parser.add_argument("--workers", type=int, default=4, metavar="N", help="dataloader threads")
    parser.add_argument("--base-size", type=int, default=512, help="base image size")
    parser.add_argument("--crop-size", type=int, default=512, help="crop or padded image size")
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="pad",
        choices=["pad", "crop", "resize", "none"],
        help="how jijie images are normalized to a consistent batch size",
    )
    parser.add_argument(
        "--split-profile",
        type=str,
        default="annotated",
        choices=["annotated", "all_usable"],
        help="which jijie split manifest set to use",
    )
    parser.add_argument(
        "--selected-classes",
        type=str,
        default=None,
        help="comma-separated original jijie class ids to keep, e.g. 1,2,3",
    )
    parser.add_argument("--sync-bn", type=bool, default=None, help="whether to use sync batch norm")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="freeze batch norm parameters")
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"], help="loss function type")
    parser.add_argument("--epochs", type=int, default=None, metavar="N", help="number of epochs to train")
    parser.add_argument("--start_epoch", type=int, default=0, metavar="N", help="manual start epoch")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N", help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=None, metavar="N", help="evaluation batch size")
    parser.add_argument(
        "--use-balanced-weights",
        action="store_true",
        default=False,
        help="use class balanced weights",
    )
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=["poly", "step", "cos"],
        help="learning rate scheduler",
    )
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, metavar="M", help="weight decay")
    parser.add_argument("--nesterov", action="store_true", default=False, help="use Nesterov momentum")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma-separated GPU ids")
    parser.add_argument("--seed", type=int, default=3407, metavar="S", help="random seed")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--checkname", type=str, default=None, help="checkpoint directory name")
    parser.add_argument("--ft", action="store_true", default=False, help="finetune from a checkpoint")
    parser.add_argument("--eval-interval", type=int, default=1, help="validation interval")
    parser.add_argument("--no-val", action="store_true", default=False, help="skip validation")
    return parser


def apply_runtime_defaults(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        print("Using GPU ids: {}".format(args.gpu_ids))
    else:
        args.gpu_ids = []
        print("CUDA is disabled or unavailable. Training will run on CPU.")

    if args.sync_bn is None:
        args.sync_bn = args.cuda and len(args.gpu_ids) > 1

    if args.epochs is None:
        default_epochs = {
            "coco": 30,
            "cityscapes": 200,
            "pascal": 50,
            "jijie": JIJIE_DEFAULT_EPOCHS,
        }
        if args.dataset not in default_epochs:
            raise KeyError("Please provide --epochs for dataset '{}'".format(args.dataset))
        args.epochs = default_epochs[args.dataset]

    if args.batch_size is None:
        gpu_count = max(1, len(args.gpu_ids))
        args.batch_size = 4 * gpu_count

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        default_lrs = {
            "coco": 0.1,
            "cityscapes": 0.01,
            "pascal": 0.007,
            "jijie": JIJIE_DEFAULT_LR,
        }
        if args.dataset not in default_lrs:
            raise KeyError("Please provide --lr for dataset '{}'".format(args.dataset))

        if args.dataset == "jijie":
            args.lr = default_lrs[args.dataset]
        else:
            gpu_count = max(1, len(args.gpu_ids))
            args.lr = default_lrs[args.dataset] / (4 * gpu_count) * args.batch_size

    if args.checkname is None:
        args.checkname = "deeplab-{}".format(args.backbone)

    args.selected_classes = parse_selected_classes(args.selected_classes)


def set_random_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_runtime_defaults(args)
    set_random_seed(args.seed, args.cuda)

    print(args)
    trainer = Trainer(args)
    print("Starting Epoch:", trainer.args.start_epoch)
    print("Total Epochs:", trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == args.eval_interval - 1:
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
