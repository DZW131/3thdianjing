import argparse
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from mypath import Path
from utils.calculate_weights import calculate_weigths_labels
from utils.checkpoint import load_checkpoint, state_dict_from_model
from utils.jijie import apply_json_config_overrides, finalize_jijie_args, summarize_jijie_run
from utils.jijie.quantify import quantify_task_prediction
from utils.loss import MetaSSLLabeledLoss, SegmentationLosses
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
    ranked_metrics = [(class_id, float(value)) for class_id, value in enumerate(metric_values) if value > 0]
    if not ranked_metrics:
        return "none"
    ranked_metrics.sort(key=lambda item: item[1], reverse=True)
    return ", ".join("c{}={:.4f}".format(class_id, value) for class_id, value in ranked_metrics[:max_items])


def aggregate_numeric_dicts(rows):
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if values:
            summary[key] = float(np.mean(values))
    return summary


@contextmanager
def batchnorm_eval_only(model):
    """Temporarily freeze BatchNorm statistics while leaving other modules unchanged."""
    bn_modules = []
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, SynchronizedBatchNorm2d)):
            bn_modules.append((module, module.training))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in bn_modules:
            module.train(was_training)


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
        self.has_validation = len(self.val_loader.dataset) > 0
        if not self.has_validation and not args.no_val:
            print("[Run] validation split is empty; switching to no-val mode for this run.")
            args.no_val = True

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
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + "_classes_weights.npy")
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.metassl_labeled_criterion = None
        if getattr(args, "enable_metassl_labeled", False):
            if args.loss_type != "ce_dice":
                print("[MetaSSL] labeled heterogeneous loss is implemented as CE+Dice; base loss is {}.".format(args.loss_type))
            self.metassl_labeled_criterion = MetaSSLLabeledLoss(
                num_classes=model.nclass if hasattr(model, "nclass") else self.nclass,
                class_weight=weight.cuda() if (weight is not None and args.cuda) else weight,
                beta=args.metassl_beta,
                delta_l=args.metassl_delta_l,
                ema_alpha=args.metassl_ema_alpha,
                initial_threshold=args.metassl_initial_threshold,
                min_threshold=args.metassl_min_threshold,
                max_threshold=args.metassl_max_threshold,
                min_region_weight=args.metassl_min_region_weight,
                protect_positive_labels=args.metassl_protect_positive_labels,
                positive_label_min_weight=args.metassl_positive_label_min_weight,
            )
        self.model = model
        self.optimizer = optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader), verbose=False)

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = -1.0
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
            self.best_pred = checkpoint.get("best_pred", checkpoint.get("selection_score", -1.0))
            print("[Checkpoint] loaded '{}' (epoch {})".format(args.resume, args.start_epoch))

        if args.ft:
            args.start_epoch = 0

    def print_run_overview(self):
        test_samples = len(self.test_loader.dataset) if self.test_loader is not None else 0
        print(
            "[Run] dataset={} task={} backbone={} classes={} train/val/test={}/{}/{}".format(
                self.args.dataset,
                getattr(self.args, "task_name", "default"),
                self.args.backbone,
                self.nclass,
                len(self.train_loader.dataset),
                len(self.val_loader.dataset),
                test_samples,
            )
        )
        print(
            "[Run] epochs={} batch={} lr={:.6f} train_resize={} eval_resize={} crop={} workers={} gpus={}".format(
                self.args.epochs,
                self.args.batch_size,
                self.args.lr,
                self.args.train_resize_mode,
                self.args.eval_resize_mode,
                self.args.crop_size,
                self.args.workers,
                self.args.gpu_ids,
            )
        )
        print(
            "[Run] checkname={} loss={} selection_metric={} outputs={}".format(
                self.args.checkname,
                self.args.loss_type,
                "final_epoch" if self.args.no_val else self.args.selection_metric,
                self.saver.experiment_dir,
            )
        )
        for line in summarize_jijie_run(self.args):
            print(line)
        if self.args.batch_size == 1:
            print("[Run] batch_size=1 detected; BatchNorm layers will be kept in eval mode during training.")
        if getattr(self.args, "enable_metassl_labeled", False):
            print(
                "[MetaSSL] labeled=on weight={} warmup={} beta={} delta_l={} ema_alpha={} min_region_weight={} protect_positive={} positive_min={}".format(
                    self.args.metassl_labeled_weight,
                    self.args.metassl_labeled_warmup_epochs,
                    self.args.metassl_beta,
                    self.args.metassl_delta_l,
                    self.args.metassl_ema_alpha,
                    self.args.metassl_min_region_weight,
                    self.args.metassl_protect_positive_labels,
                    self.args.metassl_positive_label_min_weight,
                )
            )

    def _set_batchnorm_eval(self):
        target_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        for module in target_model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                module.eval()

    def _prior_scale(self, epoch, weight_attr, warmup_attr):
        base_weight = float(getattr(self.args, weight_attr, 0.0) or 0.0)
        if base_weight <= 0:
            return 0.0
        warmup_epochs = int(getattr(self.args, warmup_attr, 0) or 0)
        if warmup_epochs <= 0:
            return base_weight
        return base_weight * min(1.0, float(epoch + 1) / float(warmup_epochs))

    def _metassl_labeled_blend(self, epoch):
        if not getattr(self.args, "enable_metassl_labeled", False):
            return 0.0
        max_weight = float(getattr(self.args, "metassl_labeled_weight", 1.0) or 0.0)
        if max_weight <= 0:
            return 0.0
        warmup_epochs = int(getattr(self.args, "metassl_labeled_warmup_epochs", 0) or 0)
        if warmup_epochs <= 0:
            return min(1.0, max_weight)
        return min(1.0, max_weight * min(1.0, float(epoch + 1) / float(warmup_epochs)))

    def _auxiliary_prior_loss(
        self,
        output,
        sample,
        epoch,
        enabled_attr,
        target_class_attr,
        label_key,
        weight_key,
        loss_weight_attr,
        warmup_attr,
    ):
        if not getattr(self.args, enabled_attr, False):
            return output.new_tensor(0.0)
        if label_key not in sample or weight_key not in sample:
            return output.new_tensor(0.0)

        target_class_id = getattr(self.args, target_class_attr, None)
        if target_class_id is None or target_class_id >= output.size(1):
            return output.new_tensor(0.0)

        scale = self._prior_scale(epoch, loss_weight_attr, warmup_attr)
        if scale <= 0:
            return output.new_tensor(0.0)

        prior_label = sample[label_key].to(device=output.device, dtype=output.dtype)
        prior_weight = sample[weight_key].to(device=output.device, dtype=output.dtype)
        positive_weight = prior_weight * (prior_label > 0.5).to(dtype=output.dtype)
        normalizer = positive_weight.sum()
        if normalizer.item() <= 0:
            return output.new_tensor(0.0)

        log_probs = F.log_softmax(output, dim=1)
        class_log_prob = log_probs[:, int(target_class_id), :, :]
        return -scale * (class_log_prob * positive_weight).sum() / torch.clamp(normalizer, min=1.0)

    def _t_tubule_prior_loss(self, output, sample, epoch):
        return self._auxiliary_prior_loss(
            output,
            sample,
            epoch,
            enabled_attr="enable_t_tubule_prior",
            target_class_attr="t_tubule_prior_target_class_id",
            label_key="t_tubule_prior_label",
            weight_key="t_tubule_prior_weight",
            loss_weight_attr="t_tubule_prior_loss_weight",
            warmup_attr="t_tubule_prior_warmup_epochs",
        )

    def _side_tubule_prior_loss(self, output, sample, epoch):
        return self._auxiliary_prior_loss(
            output,
            sample,
            epoch,
            enabled_attr="enable_side_tubule_prior",
            target_class_attr="side_tubule_prior_target_class_id",
            label_key="side_tubule_prior_label",
            weight_key="side_tubule_prior_weight",
            loss_weight_attr="side_tubule_prior_loss_weight",
            warmup_attr="side_tubule_prior_warmup_epochs",
        )

    def training(self, epoch):
        train_loss = 0.0
        epoch_start = time.time()
        self.model.train()
        if self.args.freeze_bn or self.args.batch_size == 1:
            self._set_batchnorm_eval()
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

            self.scheduler(self.optimizer, i, epoch, max(0.0, self.best_pred))
            self.optimizer.zero_grad()
            output = self.model(image)
            base_seg_loss = self.criterion(output, target)
            seg_loss = base_seg_loss
            metassl_stats = None
            metassl_blend = self._metassl_labeled_blend(epoch)
            if self.metassl_labeled_criterion is not None and metassl_blend > 0:
                with torch.no_grad(), batchnorm_eval_only(self.model):
                    ref_output = self.model(image)
                metassl_loss, metassl_stats = self.metassl_labeled_criterion(output, ref_output, target)
                seg_loss = (1.0 - metassl_blend) * base_seg_loss + metassl_blend * metassl_loss
            t_tubule_prior_loss = self._t_tubule_prior_loss(output, sample, epoch)
            side_tubule_prior_loss = self._side_tubule_prior_loss(output, sample, epoch)
            prior_loss = t_tubule_prior_loss + side_tubule_prior_loss
            loss = seg_loss + prior_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            average_loss = train_loss / (i + 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.show_progress:
                tbar.set_postfix(loss="{:.4f}".format(average_loss), lr="{:.6f}".format(current_lr), refresh=False)

            global_step = i + num_batches * epoch
            self.writer.add_scalar("train/total_loss_iter", loss.item(), global_step)
            if metassl_stats is not None:
                self.writer.add_scalar("train/base_seg_loss_iter", base_seg_loss.item(), global_step)
                self.writer.add_scalar("train/metassl_blend_iter", metassl_blend, global_step)
                self.writer.add_scalar("train/metassl_threshold_mean_iter", metassl_stats["metassl_threshold_mean"], global_step)
                self.writer.add_scalar("train/metassl_uc_fraction_iter", metassl_stats["metassl_uc_fraction"], global_step)
                self.writer.add_scalar("train/metassl_dc_fraction_iter", metassl_stats["metassl_dc_fraction"], global_step)
                self.writer.add_scalar(
                    "train/metassl_positive_disagreement_fraction_iter",
                    metassl_stats["metassl_positive_disagreement_fraction"],
                    global_step,
                )
            if t_tubule_prior_loss.item() > 0:
                self.writer.add_scalar("train/t_tubule_prior_loss_iter", t_tubule_prior_loss.item(), global_step)
            if side_tubule_prior_loss.item() > 0:
                self.writer.add_scalar("train/side_tubule_prior_loss_iter", side_tubule_prior_loss.item(), global_step)

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
            selection_score = float(epoch + 1)
            is_final_epoch = (epoch + 1) == self.args.epochs
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": state_dict_from_model(self.model),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": selection_score,
                    "selection_score": selection_score,
                    "selection_metric_name": "final_epoch",
                },
                is_best=False,
                filename="checkpoint_last.pth.tar",
                force_best=is_final_epoch,
            )
            if is_final_epoch:
                self.best_pred = selection_score
                print(
                    "[Epoch {:03d}/{:03d}] no-val final checkpoint promoted to model_best.pth.tar".format(
                        epoch + 1,
                        self.args.epochs,
                    )
                )

    def _select_validation_score(self, metric_summary):
        metric_name = self.args.selection_metric
        if metric_name == "mean_positive_dice":
            return metric_summary.get("mean_positive_dice", metric_summary["mean_iou"])
        if metric_name in {"mean_positive_iou", "mean_target_iou"}:
            return metric_summary.get("mean_positive_iou", metric_summary["mean_iou"])
        return metric_summary["mean_iou"]

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
        val_loss = 0.0
        image_level_rows = []

        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)
            val_loss += loss.item()
            if self.show_progress:
                average_loss = val_loss / (i + 1)
                tbar.set_postfix(loss="{:.4f}".format(average_loss), refresh=False)

            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            self.evaluator.add_batch(target_np, pred)

            if self.args.dataset == "jijie" and getattr(self.args, "quantify_class_ids", None):
                batch_size = pred.shape[0]
                for batch_index in range(batch_size):
                    sample_id = sample.get("sample_id", ["sample"])[batch_index]
                    image_summary, _, _ = quantify_task_prediction(
                        getattr(self.args, "task_name", "jijie"),
                        gt_mask=target_np[batch_index].astype(np.int32),
                        pred_mask=pred[batch_index].astype(np.int32),
                        class_names=self.val_loader.dataset.class_names,
                        quantify_class_ids=self.args.quantify_class_ids,
                        sample_id=sample_id,
                    )
                    image_level_rows.append(image_summary)

        average_val_loss = val_loss / max(1, len(self.val_loader))
        per_class_iou = self.evaluator.per_class_iou()
        per_class_dice = self.evaluator.per_class_dice()
        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mean_positive_iou = self.evaluator.mean_over_classes(per_class_iou, self.args.metric_target_class_ids)
        mean_positive_dice = self.evaluator.mean_over_classes(per_class_dice, self.args.metric_target_class_ids)
        top_iou_summary = format_top_class_metrics(per_class_iou)
        image_metric_summary = aggregate_numeric_dicts(image_level_rows)

        metric_summary = {
            "pixel_accuracy": acc,
            "pixel_accuracy_class": acc_class,
            "mean_iou": miou,
            "frequency_weighted_iou": fwiou,
            "mean_positive_iou": mean_positive_iou,
            "mean_positive_dice": mean_positive_dice,
        }
        metric_summary.update(image_metric_summary)
        selection_score = self._select_validation_score(metric_summary)

        self.writer.add_scalar("val/total_loss_epoch", average_val_loss, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)
        self.writer.add_scalar("val/Acc", acc, epoch)
        self.writer.add_scalar("val/Acc_class", acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", fwiou, epoch)
        self.writer.add_scalar("val/mean_positive_iou", mean_positive_iou, epoch)
        self.writer.add_scalar("val/mean_positive_dice", mean_positive_dice, epoch)
        self.writer.add_scalar("val/selection_score", selection_score, epoch)
        for key, value in image_metric_summary.items():
            self.writer.add_scalar("val_task/{}".format(key), value, epoch)

        is_best = selection_score > self.best_pred
        if is_best:
            self.best_pred = selection_score

        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict_from_model(self.model),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
                "selection_score": selection_score,
                "selection_metric_name": self.args.selection_metric,
            },
            is_best=is_best,
        )

        print(
            "[Epoch {:03d}/{:03d}] val loss={:.4f} acc={:.4f} acc_cls={:.4f} mIoU={:.4f} posIoU={:.4f} posDice={:.4f} score({})={:.4f} best={:.4f} time={}".format(
                epoch + 1,
                self.args.epochs,
                average_val_loss,
                acc,
                acc_class,
                miou,
                mean_positive_iou,
                mean_positive_dice,
                self.args.selection_metric,
                selection_score,
                self.best_pred,
                format_duration(time.time() - epoch_start),
            )
        )
        if top_iou_summary != "none":
            print("  per-class IoU>0: {}".format(top_iou_summary))
        if image_metric_summary:
            compact = ", ".join("{}={:.4f}".format(key, value) for key, value in sorted(image_metric_summary.items())[:6])
            print("  task-metrics: {}".format(compact))


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Training")
    parser.add_argument("--config", type=str, default=None, help="optional JSON config file")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"], help="backbone name")
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
    parser.add_argument("--task-name", type=str, default=None, help="logical jijie task name, e.g. mito / mito_sr / sarcomere")
    parser.add_argument("--manifest-dir", type=str, default=None, help="directory that contains train.txt / val.txt / test.txt manifests")
    parser.add_argument("--use-sbd", action="store_true", default=True, help="use SBD dataset for Pascal")
    parser.add_argument("--workers", type=int, default=4, metavar="N", help="dataloader threads")
    parser.add_argument("--base-size", type=int, default=512, help="base image size")
    parser.add_argument("--crop-size", type=int, default=512, help="crop or padded image size")
    parser.add_argument("--resize-mode", type=str, default="pad", choices=["pad", "crop", "resize", "none", "random_crop"], help="legacy shared resize mode")
    parser.add_argument("--train-resize-mode", type=str, default=None, choices=["pad", "crop", "resize", "none", "random_crop"], help="train-time resize mode")
    parser.add_argument("--eval-resize-mode", type=str, default=None, choices=["pad", "crop", "resize", "none", "random_crop"], help="eval-time resize mode")
    parser.add_argument("--split-profile", type=str, default="annotated", choices=["annotated", "all_usable"], help="which jijie split manifest set to use")
    parser.add_argument("--selected-classes", type=str, default=None, help="comma-separated original jijie class ids to keep, e.g. 1,2,3")
    parser.add_argument("--metric-target-original-classes", type=str, default=None, help="comma-separated original class ids used for model selection")
    parser.add_argument("--quantify-original-classes", type=str, default=None, help="comma-separated original class ids used for image-level quantification")
    parser.add_argument("--label-dilate-original-classes", type=str, default=None, help="comma-separated original class ids to dilate on training masks")
    parser.add_argument("--label-dilate-radius", type=int, default=0, help="dilation radius for selected training labels")
    parser.add_argument("--enable-t-tubule-prior", action="store_true", default=False, help="enable sarcomere T-tubule pseudo-line auxiliary supervision")
    parser.add_argument("--t-tubule-prior-z-original-classes", type=str, default=None, help="original class ids used as Z-line anchors")
    parser.add_argument("--t-tubule-prior-m-original-classes", type=str, default=None, help="original class ids used as M-line anchors")
    parser.add_argument("--t-tubule-prior-target-original-class", type=str, default=None, help="original class id supervised by the pseudo-line prior")
    parser.add_argument("--t-tubule-prior-loss-weight", type=float, default=0.0, help="maximum auxiliary prior loss weight")
    parser.add_argument("--t-tubule-prior-warmup-epochs", type=int, default=0, help="epochs used to ramp the auxiliary prior loss")
    parser.add_argument("--t-tubule-prior-line-radius", type=int, default=1, help="pixel dilation radius for generated pseudo-lines")
    parser.add_argument("--t-tubule-prior-min-distance", type=float, default=8.0, help="minimum Z/M endpoint distance for a pseudo-line")
    parser.add_argument("--t-tubule-prior-max-distance", type=float, default=384.0, help="maximum Z/M endpoint distance for a pseudo-line")
    parser.add_argument("--t-tubule-prior-min-component-area", type=int, default=4, help="minimum Z/M component size used for endpoint extraction")
    parser.add_argument("--t-tubule-prior-min-confidence", type=float, default=0.25, help="minimum local confidence for generated pseudo-lines")
    parser.add_argument("--t-tubule-prior-exclude-non-background", type=bool, default=True, help="avoid adding prior pixels over non-background non-target labels")
    parser.add_argument("--train-vertical-flip", action="store_true", default=False, help="enable random vertical flip in training")
    parser.add_argument("--train-rotate-degree", type=float, default=0.0, help="small-angle random rotation for training")
    parser.add_argument("--sync-bn", type=bool, default=None, help="whether to use sync batch norm")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="freeze batch norm parameters")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal", "dice", "ce_dice", "focal_dice", "tversky_dice"],
        help="loss function type",
    )
    parser.add_argument("--selection-metric", type=str, default=None, choices=["mean_iou", "mean_positive_iou", "mean_positive_dice", "mean_target_iou"], help="metric used to keep the best checkpoint")
    parser.add_argument("--inference-mode", type=str, default="direct", choices=["direct", "sliding"], help="stored in config for consistency with evaluation")
    parser.add_argument("--sliding-window-size", type=int, default=None, help="stored in config for consistency with evaluation")
    parser.add_argument("--sliding-window-stride", type=int, default=None, help="stored in config for consistency with evaluation")
    parser.add_argument("--epochs", type=int, default=None, metavar="N", help="number of epochs to train")
    parser.add_argument("--start_epoch", type=int, default=0, metavar="N", help="manual start epoch")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N", help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=None, metavar="N", help="evaluation batch size")
    parser.add_argument("--use-balanced-weights", action="store_true", default=False, help="use class balanced weights")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate")
    parser.add_argument("--lr-scheduler", type=str, default="poly", choices=["poly", "step", "cos"], help="learning rate scheduler")
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
    parser.add_argument("--enable-metassl-labeled", action="store_true", default=False, help="enable labeled-image MetaSSL heterogeneous CE+Dice loss")
    parser.add_argument("--metassl-labeled-weight", type=float, default=1.0, help="maximum blend ratio for MetaSSL labeled loss")
    parser.add_argument("--metassl-labeled-warmup-epochs", type=int, default=20, help="epochs used to ramp MetaSSL labeled loss")
    parser.add_argument("--metassl-beta", type=float, default=3.0, help="generalized Gaussian beta for MetaSSL region weights")
    parser.add_argument("--metassl-delta-l", type=float, default=0.6, help="labeled-image interval delta_l for MetaSSL region weights")
    parser.add_argument("--metassl-ema-alpha", type=float, default=0.99, help="EMA alpha for adaptive MetaSSL confidence threshold")
    parser.add_argument("--metassl-initial-threshold", type=float, default=0.5, help="initial class-wise confidence threshold")
    parser.add_argument("--metassl-min-threshold", type=float, default=0.5, help="minimum adaptive confidence threshold")
    parser.add_argument("--metassl-max-threshold", type=float, default=0.99, help="maximum adaptive confidence threshold")
    parser.add_argument("--metassl-min-region-weight", type=float, default=0.05, help="minimum spatial weight assigned by MetaSSL")
    parser.add_argument("--metassl-protect-positive-labels", action="store_true", default=False, help="avoid heavily down-weighting manually labeled positive pixels")
    parser.add_argument("--metassl-positive-label-min-weight", type=float, default=0.8, help="minimum weight for positive-label pixels when predictions disagree")
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
        if args.dataset == "jijie" and getattr(args, "task_name", None):
            args.checkname = "deeplab-{}-{}".format(args.backbone, args.task_name)
        else:
            args.checkname = "deeplab-{}".format(args.backbone)


def set_random_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_json_config_overrides(parser, args)
    args = finalize_jijie_args(args)
    apply_runtime_defaults(args)
    set_random_seed(args.seed, args.cuda)

    trainer = Trainer(args)
    trainer.print_run_overview()
    print("[Run] start_epoch={} total_epochs={}".format(trainer.args.start_epoch, trainer.args.epochs))

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if trainer.has_validation and not trainer.args.no_val and epoch % args.eval_interval == args.eval_interval - 1:
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
