import argparse
import os
import random
import sys
import time

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


def format_duration(seconds):
    total_seconds = int(round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return "{:d}h{:02d}m{:02d}s".format(hours, minutes, seconds)
    if minutes:
        return "{:d}m{:02d}s".format(minutes, seconds)
    return "{:d}s".format(seconds)


def format_top_class_metrics(metric_values, max_items=5):
    ranked_metrics = [
        (class_id, float(value))
        for class_id, value in enumerate(metric_values)
        if value > 0
    ]
    if not ranked_metrics:
        return "none"

    ranked_metrics.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(
        "c{}={:.4f}".format(class_id, value)
        for class_id, value in ranked_metrics[:max_items]
    )


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
        self.show_progress = sys.stdout.isatty() and not args.no_progress

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
        self.scheduler = LR_Scheduler(
            args.lr_scheduler,
            args.lr,
            args.epochs,
            len(self.train_loader),
            verbose=False,
        )

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
            print("[Checkpoint] loaded '{}' (epoch {})".format(args.resume, args.start_epoch))

        if args.ft:
            args.start_epoch = 0

    def print_run_overview(self):
        test_samples = len(self.test_loader.dataset) if self.test_loader is not None else 0
        selected_classes = (
            ",".join(str(class_id) for class_id in self.args.selected_classes)
            if self.args.selected_classes
            else "all"
        )
        print(
            "[Run] dataset={} backbone={} classes={} train/val/test={}/{}/{}".format(
                self.args.dataset,
                self.args.backbone,
                self.nclass,
                len(self.train_loader.dataset),
                len(self.val_loader.dataset),
                test_samples,
            )
        )
        print(
            "[Run] epochs={} batch={} lr={:.6f} resize={} crop={} split={} workers={} gpus={}".format(
                self.args.epochs,
                self.args.batch_size,
                self.args.lr,
                self.args.resize_mode,
                self.args.crop_size,
                self.args.split_profile,
                self.args.workers,
                self.args.gpu_ids,
            )
        )
        print(
            "[Run] checkname={} selected_classes={} progress_bars={} outputs={}".format(
                self.args.checkname,
                selected_classes,
                "on" if self.show_progress else "off",
                self.saver.experiment_dir,
            )
        )

    def training(self, epoch):
        train_loss = 0.0
        epoch_start = time.time()
        self.model.train()
        num_batches = len(self.train_loader)
        vis_interval = max(1, num_batches // 10)
        tbar = tqdm(
            self.train_loader,
            desc="Train {:03d}/{:03d}".format(epoch + 1, self.args.epochs),
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )

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
            average_loss = train_loss / (i + 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.show_progress:
                tbar.set_postfix(loss="{:.4f}".format(average_loss), lr="{:.6f}".format(current_lr), refresh=False)

            global_step = i + num_batches * epoch
            self.writer.add_scalar("train/total_loss_iter", loss.item(), global_step)

            if i % vis_interval == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        average_train_loss = train_loss / max(1, num_batches)
        self.writer.add_scalar("train/total_loss_epoch", average_train_loss, epoch)
        print(
            "[Epoch {:03d}/{:03d}] train loss={:.4f} lr={:.6f} steps={} samples={} time={}".format(
                epoch + 1,
                self.args.epochs,
                average_train_loss,
                self.optimizer.param_groups[0]["lr"],
                num_batches,
                len(self.train_loader.dataset),
                format_duration(time.time() - epoch_start),
            )
        )

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
        epoch_start = time.time()
        tbar = tqdm(
            self.val_loader,
            desc="Val   {:03d}/{:03d}".format(epoch + 1, self.args.epochs),
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )
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
            if self.show_progress:
                average_loss = test_loss / (i + 1)
                tbar.set_postfix(loss="{:.4f}".format(average_loss), refresh=False)

            pred = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target_np, pred)

        average_val_loss = test_loss / max(1, len(self.val_loader))
        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        per_class_iou = self.evaluator.per_class_iou()
        valid_class_count = int(self.evaluator.valid_class_mask().sum())
        top_iou_summary = format_top_class_metrics(per_class_iou)

        self.writer.add_scalar("val/total_loss_epoch", average_val_loss, epoch)
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

        print(
            "[Epoch {:03d}/{:03d}] val   loss={:.4f} acc={:.4f} acc_cls={:.4f} mIoU={:.4f} fwIoU={:.4f} best={:.4f} valid_cls={}/{} time={}".format(
                epoch + 1,
                self.args.epochs,
                average_val_loss,
                acc,
                acc_class,
                miou,
                fwiou,
                self.best_pred,
                valid_class_count,
                self.nclass,
                format_duration(time.time() - epoch_start),
            )
        )
        if top_iou_summary != "none":
            print("  per-class IoU>0: {}".format(top_iou_summary))


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
    parser.add_argument("--no-progress", action="store_true", default=False, help="disable tqdm progress bars")
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

    trainer = Trainer(args)
    trainer.print_run_overview()
    print("[Run] start_epoch={} total_epochs={}".format(trainer.args.start_epoch, trainer.args.epochs))

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == args.eval_interval - 1:
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
