import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from utils.checkpoint import load_checkpoint
from utils.jijie import apply_json_config_overrides, finalize_jijie_args, summarize_jijie_run
from utils.jijie.inference import predict_logits
from utils.jijie.partial_label_metrics import aggregate_gt_overlap_rows, compute_gt_overlap_rows
from utils.jijie.postprocess import postprocess_mito_mask, DEFAULT_CLASSIFIER_PATH
from utils.jijie.quantify import quantify_task_prediction
from utils.jijie.visualization import (
    blend_mask,
    colorize_mask,
    palette_rows,
    render_class_legend,
    to_display_class_names,
)
from utils.metrics import Evaluator

try:
    import seaborn as sns
except ImportError:
    sns = None


def auto_resume_path(dataset_name):
    run_root = Path("run") / dataset_name
    if not run_root.is_dir():
        return None
    candidates = sorted(run_root.glob("*/model_best.pth.tar"))
    if not candidates:
        candidates = sorted(run_root.glob("*/checkpoint_last.pth.tar"))
    if not candidates:
        return None
    return str(candidates[-1])


def denormalize_image(image_tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image_tensor.numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_numeric_rows(rows):
    if not rows:
        return {}
    summary = {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def aggregate_class_rows(rows):
    grouped = {}
    for row in rows:
        key = (row.get("source"), row.get("class_id"), row.get("class_name"))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (source, class_id, class_name), group_rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        summary = {
            "source": source,
            "class_id": class_id,
            "class_name": class_name,
            "sample_count": len(group_rows),
        }
        keys = sorted(set().union(*(row.keys() for row in group_rows)))
        for key in keys:
            values = [
                row[key]
                for row in group_rows
                if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
            ]
            if not values:
                continue
            summary[key + "_mean"] = float(np.mean(values))
            if key in {"component_count", "area_px_sum", "area_um2_sum"}:
                summary[key + "_sum"] = float(np.sum(values))
        summary_rows.append(summary)
    return summary_rows


def parse_int_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


class SegmentationEvaluator(object):
    def __init__(self, args):
        self.args = args
        ensure_dir(args.save_dir)

        loader_kwargs = {"num_workers": args.workers, "pin_memory": args.cuda}
        train_loader, val_loader, test_loader, self.nclass = make_data_loader(args, **loader_kwargs)

        if args.split == "train":
            self.loader = train_loader
        elif args.split == "val":
            self.loader = val_loader
        else:
            self.loader = test_loader or val_loader
            if test_loader is None:
                print("Test split is unavailable. Falling back to validation split.")

        self.class_names = getattr(self.loader.dataset, "class_names", [f"Class_{i}" for i in range(self.nclass)])
        self.display_class_names = to_display_class_names(self.class_names, language="en")
        self.model = DeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=False,
            freeze_bn=args.freeze_bn,
        )
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        checkpoint = load_checkpoint(args.resume, self.model, map_location="cpu")
        self.best_pred = checkpoint.get("best_pred", checkpoint.get("selection_score", 0.0))
        if args.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.evaluator = Evaluator(self.nclass)
        if self.args.partial_label_class_ids is None:
            fallback_class_ids = self.args.quantify_class_ids or self.args.metric_target_class_ids
            if fallback_class_ids:
                self.args.partial_label_class_ids = [int(class_id) for class_id in fallback_class_ids]
            else:
                self.args.partial_label_class_ids = list(range(1, self.nclass))

    def _load_gray_image(self, sample, batch_index):
        image_path = sample.get("image_path", None)
        if image_path is None:
            return None
        if isinstance(image_path, (list, tuple)):
            path = image_path[batch_index]
        else:
            path = image_path
        if not os.path.isfile(path):
            return None
        img = Image.open(path).convert("L")
        original_size = sample.get("original_size", None)
        if original_size is not None:
            if isinstance(original_size, (list, tuple)):
                os_h, os_w = original_size[batch_index] if isinstance(original_size[0], (list, tuple)) else original_size
            else:
                os_h, os_w = original_size
            if img.size != (os_w, os_h):
                img = img.resize((os_w, os_h), Image.BILINEAR)
        return np.asarray(img)

    def run(self):
        self.evaluator.reset()
        vis_samples = []
        image_rows = []
        object_rows = []
        detection_rows = []
        class_rows = []
        partial_label_rows = []
        postprocess_stats = []

        tbar = tqdm(self.loader, desc="Evaluating", disable=not (self.args.show_progress))
        with torch.no_grad():
            for sample in tbar:
                image = sample["image"]
                target = sample["label"]
                if self.args.cuda:
                    image = image.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                logits = predict_logits(
                    self.model,
                    image,
                    num_classes=self.nclass,
                    inference_mode=self.args.inference_mode,
                    window_size=self.args.sliding_window_size,
                    stride=self.args.sliding_window_stride,
                )
                pred = torch.argmax(logits, dim=1)

                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()

                if self.args.postprocess_mito and getattr(self.args, "task_name", None) == "mito":
                    batch_size = pred_np.shape[0]
                    for batch_index in range(batch_size):
                        gray_image = self._load_gray_image(sample, batch_index)
                        if gray_image is None:
                            continue
                        post_pred, post_stats = postprocess_mito_mask(
                            pred_np[batch_index].astype(np.int32),
                            gray_image=gray_image,
                            classifier_path=self.args.postprocess_classifier,
                            use_gray_tiebreak=True,
                            use_texture_refine=bool(self.args.postprocess_classifier),
                        )
                        pred_np[batch_index] = post_pred
                        postprocess_stats.append(post_stats)

                self.evaluator.add_batch(target_np, pred_np)

                if self.args.partial_label_eval:
                    batch_size = pred_np.shape[0]
                    for batch_index in range(batch_size):
                        sample_id = sample.get("sample_id", ["sample"])[batch_index]
                        partial_label_rows.extend(
                            compute_gt_overlap_rows(
                                gt_mask=target_np[batch_index].astype(np.int32),
                                pred_mask=pred_np[batch_index].astype(np.int32),
                                class_ids=self.args.partial_label_class_ids,
                                class_names=self.class_names,
                                sample_id=sample_id,
                                min_overlap_pixels=self.args.partial_label_overlap_min_pixels,
                            )
                        )

                if self.args.quantify_class_ids:
                    batch_size = pred_np.shape[0]
                    for batch_index in range(batch_size):
                        sample_id = sample.get("sample_id", ["sample"])[batch_index]
                        image_summary, sample_object_rows, sample_detection_rows, sample_class_rows = quantify_task_prediction(
                            task_name=getattr(self.args, "task_name", "jijie"),
                            gt_mask=target_np[batch_index].astype(np.int32),
                            pred_mask=pred_np[batch_index].astype(np.int32),
                            class_names=self.class_names,
                            quantify_class_ids=self.args.quantify_class_ids,
                            sample_id=sample_id,
                            um_per_pixel=self.args.um_per_pixel,
                            return_class_rows=True,
                        )
                        image_rows.append(image_summary)
                        object_rows.extend(sample_object_rows)
                        detection_rows.extend(sample_detection_rows)
                        class_rows.extend(sample_class_rows)

                if self.args.visualize and len(vis_samples) < self.args.num_vis_samples:
                    cpu_images = image.detach().cpu()
                    cpu_targets = target.detach().cpu()
                    cpu_preds = pred.detach().cpu()
                    for index in range(cpu_images.size(0)):
                        if len(vis_samples) >= self.args.num_vis_samples:
                            break
                        vis_samples.append(
                            {
                                "image": cpu_images[index].clone(),
                                "target": cpu_targets[index].clone(),
                                "pred": cpu_preds[index].clone(),
                                "sample_id": sample.get("sample_id", ["sample"])[index],
                            }
                        )

        summary = self._build_summary(image_rows, detection_rows, partial_label_rows)
        if postprocess_stats:
            unified_total = sum(s.get("unified_count", 0) for s in postprocess_stats)
            refined_total = sum(s.get("refined_count", 0) for s in postprocess_stats)
            summary["postprocess"] = {
                "enabled": True,
                "images_processed": len(postprocess_stats),
                "instances_unified": int(unified_total),
                "instances_refined": int(refined_total),
            }
            print("[Postprocess] unified {} instances, refined {} instances across {} images".format(
                unified_total, refined_total, len(postprocess_stats)))
        self._save_metrics(summary, image_rows, object_rows, detection_rows, class_rows, partial_label_rows)
        self._save_confusion_matrix()
        if self.args.visualize:
            self._save_visualizations(vis_samples)
        return summary

    def _build_summary(self, image_rows, detection_rows, partial_label_rows):
        confusion = self.evaluator.confusion_matrix
        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        per_class_iou = self.evaluator.per_class_iou()
        per_class_precision = self.evaluator.per_class_precision()
        per_class_recall = self.evaluator.per_class_recall()
        per_class_dice = self.evaluator.per_class_dice()

        per_class_rows = []
        for class_id in range(self.nclass):
            intersection = confusion[class_id, class_id]
            gt_total = confusion[class_id, :].sum()
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}",
                    "iou": float(per_class_iou[class_id]),
                    "dice": float(per_class_dice[class_id]),
                    "precision": float(per_class_precision[class_id]),
                    "recall": float(per_class_recall[class_id]),
                    "pixel_count": int(gt_total),
                }
            )

        image_metric_summary = aggregate_numeric_rows(image_rows)
        detection_metric_summary = aggregate_numeric_rows(detection_rows)
        partial_label_per_class_rows = []
        partial_label_overlap_summary = {}
        if self.args.partial_label_eval:
            partial_label_per_class_rows, partial_label_overlap_summary = aggregate_gt_overlap_rows(
                partial_label_rows,
                class_ids=self.args.partial_label_class_ids,
                class_names=self.class_names,
                metric_target_class_ids=self.args.metric_target_class_ids,
            )

        summary = {
            "dataset": self.args.dataset,
            "task_name": getattr(self.args, "task_name", None),
            "split": self.args.split,
            "resume": self.args.resume,
            "selection_score_from_checkpoint": float(self.best_pred),
            "pixel_accuracy": float(acc),
            "pixel_accuracy_class": float(acc_class),
            "mean_iou": float(miou),
            "frequency_weighted_iou": float(fwiou),
            "mean_positive_iou": float(self.evaluator.mean_over_classes(per_class_iou, self.args.metric_target_class_ids)),
            "mean_positive_dice": float(self.evaluator.mean_over_classes(per_class_dice, self.args.metric_target_class_ids)),
            "num_classes": int(self.nclass),
            "class_names": self.class_names,
            "per_class_metrics": per_class_rows,
            "image_metric_summary": image_metric_summary,
            "detection_metric_summary": detection_metric_summary,
            "partial_label_overlap_summary": partial_label_overlap_summary,
            "partial_label_overlap_per_class_metrics": partial_label_per_class_rows,
        }
        return summary

    def _save_metrics(self, summary, image_rows, object_rows, detection_rows, class_rows, partial_label_rows):
        write_json(os.path.join(self.args.save_dir, "metrics_summary.json"), summary)
        write_csv(
            os.path.join(self.args.save_dir, "per_class_metrics.csv"),
            summary["per_class_metrics"],
            ["class_id", "class_name", "iou", "dice", "precision", "recall", "pixel_count"],
        )
        write_csv(
            os.path.join(self.args.save_dir, "class_palette.csv"),
            palette_rows(self.class_names, self.display_class_names),
            ["class_id", "class_name", "class_name_display", "color_r", "color_g", "color_b", "hex_color"],
        )
        np.savetxt(
            os.path.join(self.args.save_dir, "confusion_matrix.csv"),
            self.evaluator.confusion_matrix,
            delimiter=",",
            fmt="%.0f",
        )

        if image_rows:
            fieldnames = sorted({key for row in image_rows for key in row.keys()})
            write_csv(os.path.join(self.args.save_dir, "image_task_metrics.csv"), image_rows, fieldnames)
        if object_rows:
            fieldnames = sorted({key for row in object_rows for key in row.keys()})
            write_csv(os.path.join(self.args.save_dir, "object_metrics.csv"), object_rows, fieldnames)
        if detection_rows:
            fieldnames = sorted({key for row in detection_rows for key in row.keys()})
            write_csv(os.path.join(self.args.save_dir, "detection_metrics.csv"), detection_rows, fieldnames)
        if class_rows:
            fieldnames = sorted({key for row in class_rows for key in row.keys()})
            write_csv(os.path.join(self.args.save_dir, "class_task_metrics.csv"), class_rows, fieldnames)
            class_summary_rows = aggregate_class_rows(class_rows)
            summary_fieldnames = sorted({key for row in class_summary_rows for key in row.keys()})
            write_csv(os.path.join(self.args.save_dir, "class_metric_summary.csv"), class_summary_rows, summary_fieldnames)
        if summary.get("partial_label_overlap_per_class_metrics"):
            fieldnames = sorted(
                {key for row in summary["partial_label_overlap_per_class_metrics"] for key in row.keys()}
            )
            write_csv(
                os.path.join(self.args.save_dir, "partial_label_overlap_per_class_metrics.csv"),
                summary["partial_label_overlap_per_class_metrics"],
                fieldnames,
            )
        if partial_label_rows:
            fieldnames = sorted({key for row in partial_label_rows for key in row.keys()})
            write_csv(
                os.path.join(self.args.save_dir, "partial_label_overlap_sample_class_metrics.csv"),
                partial_label_rows,
                fieldnames,
            )

    def _save_confusion_matrix(self):
        confusion = self.evaluator.confusion_matrix.astype(np.float64)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)

        plt.figure(figsize=(12, 10))
        if sns is not None:
            sns.heatmap(
                normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.display_class_names[: self.nclass],
                yticklabels=self.display_class_names[: self.nclass],
            )
        else:
            plt.imshow(normalized, cmap="Blues", interpolation="nearest")
            plt.colorbar(fraction=0.046, pad=0.04)
            tick_positions = np.arange(self.nclass)
            plt.xticks(tick_positions, self.display_class_names[: self.nclass], rotation=45, ha="right")
            plt.yticks(tick_positions, self.display_class_names[: self.nclass])
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _save_visualizations(self, vis_samples):
        vis_dir = ensure_dir(os.path.join(self.args.save_dir, "visualizations"))
        legend_figure = render_class_legend(
            self.class_names,
            title="Task Class Colors",
            display_class_names=self.display_class_names,
        )
        legend_figure.savefig(os.path.join(vis_dir, "class_legend.png"), dpi=200, bbox_inches="tight")
        plt.close(legend_figure)

        for sample_index, sample in enumerate(vis_samples):
            image = denormalize_image(sample["image"])
            gt_mask = sample["target"].numpy().astype(np.uint8)
            pred_mask = sample["pred"].numpy().astype(np.uint8)

            gt_colored = colorize_mask(gt_mask, self.class_names)
            pred_colored = colorize_mask(pred_mask, self.class_names)
            overlay_gt = blend_mask(image, gt_colored, alpha=0.4)
            overlay_pred = blend_mask(image, pred_colored, alpha=0.4)
            diff_mask = np.where(gt_mask == pred_mask, 0, 255).astype(np.uint8)
            diff_colored = np.stack([diff_mask, np.zeros_like(diff_mask), np.zeros_like(diff_mask)], axis=2)
            present_class_ids = sorted({int(class_id) for class_id in np.unique(np.concatenate([gt_mask.ravel(), pred_mask.ravel()]))})
            legend_fig = render_class_legend(
                self.class_names,
                present_class_ids=present_class_ids,
                title="Present Classes",
                display_class_names=self.display_class_names,
            )

            fig, axes = plt.subplots(2, 4, figsize=(22, 12))
            panels = [
                (image, "Original"),
                (gt_colored, "Ground Truth"),
                (pred_colored, "Prediction"),
                (overlay_pred, "Original + Prediction"),
                (overlay_gt, "Original + GT"),
                (diff_colored, "Prediction Error"),
            ]
            flat_axes = axes.flat
            for axis, (panel, title) in zip(flat_axes, panels):
                axis.imshow(panel)
                axis.set_title(title)
                axis.axis("off")
            legend_canvas = legend_fig.canvas
            legend_canvas.draw()
            legend_image = np.frombuffer(legend_canvas.buffer_rgba(), dtype=np.uint8)
            legend_image = legend_image.reshape(legend_canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            remaining_axes = list(axes.flat)[len(panels):]
            if remaining_axes:
                remaining_axes[0].imshow(legend_image)
                remaining_axes[0].set_title("Legend")
                remaining_axes[0].axis("off")
            for axis in remaining_axes[1:]:
                axis.axis("off")
            plt.close(legend_fig)

            plt.tight_layout()
            sample_id = sample.get("sample_id", "sample")
            prefix = os.path.join(vis_dir, f"{sample_index:03d}_{sample_id}")
            plt.savefig(prefix + "_composite.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            plt.imsave(prefix + "_gt_mask.png", gt_colored)
            plt.imsave(prefix + "_pred_mask.png", pred_colored)
            plt.imsave(prefix + "_overlay_gt.png", overlay_gt)
            plt.imsave(prefix + "_overlay_pred.png", overlay_pred)


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Evaluation")
    parser.add_argument("--config", type=str, default=None, help="optional JSON config file")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"])
    parser.add_argument("--out-stride", type=int, default=16, help="network output stride")
    parser.add_argument("--dataset", type=str, default="jijie", help="dataset name")
    parser.add_argument("--task-name", type=str, default=None, help="logical jijie task name")
    parser.add_argument("--manifest-dir", type=str, default=None, help="directory that contains train.txt / val.txt / test.txt manifests")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="dataset split")
    parser.add_argument("--workers", type=int, default=4, metavar="N", help="dataloader threads")
    parser.add_argument("--base-size", type=int, default=512, help="base image size")
    parser.add_argument("--crop-size", type=int, default=512, help="crop or padded image size")
    parser.add_argument("--resize-mode", type=str, default="pad", choices=["pad", "crop", "resize", "none", "random_crop"], help="legacy shared resize mode")
    parser.add_argument("--train-resize-mode", type=str, default=None, choices=["pad", "crop", "resize", "none", "random_crop"], help="train-time resize mode")
    parser.add_argument("--eval-resize-mode", type=str, default=None, choices=["pad", "crop", "resize", "none", "random_crop"], help="eval-time resize mode")
    parser.add_argument("--split-profile", type=str, default="annotated", choices=["annotated", "all_usable"], help="which jijie split manifest set to use")
    parser.add_argument("--selected-classes", type=str, default=None, help="comma-separated original jijie class ids to keep")
    parser.add_argument("--metric-target-original-classes", type=str, default=None, help="comma-separated original class ids for target metrics")
    parser.add_argument("--quantify-original-classes", type=str, default=None, help="comma-separated original class ids used for quantification")
    parser.add_argument("--label-dilate-original-classes", type=str, default=None, help="kept for config compatibility")
    parser.add_argument("--label-dilate-radius", type=int, default=0, help="kept for config compatibility")
    parser.add_argument("--train-vertical-flip", action="store_true", default=False, help="kept for config compatibility")
    parser.add_argument("--train-rotate-degree", type=float, default=0.0, help="kept for config compatibility")
    parser.add_argument("--sync-bn", type=bool, default=False, help="kept for CLI compatibility")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="freeze batch norm parameters")
    parser.add_argument("--selection-metric", type=str, default=None, choices=["mean_iou", "mean_positive_iou", "mean_positive_dice", "mean_target_iou"], help="kept for config compatibility")
    parser.add_argument("--inference-mode", type=str, default="direct", choices=["direct", "sliding"], help="evaluation inference mode")
    parser.add_argument("--sliding-window-size", type=int, default=None, help="sliding window size")
    parser.add_argument("--sliding-window-stride", type=int, default=None, help="sliding window stride")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N", help="training batch size placeholder")
    parser.add_argument("--test-batch-size", type=int, default=1, metavar="N", help="evaluation batch size")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma-separated GPU ids")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--save-dir", type=str, default="outputs/eval", help="directory for metrics and figures")
    parser.add_argument("--um-per-pixel", type=float, default=None, help="pixel size used for um and um2 quantification")
    parser.add_argument("--postprocess-mito", action="store_true", default=False, help="apply instance unification + texture classifier post-processing to mito task predictions")
    parser.add_argument("--postprocess-classifier", type=str, default=DEFAULT_CLASSIFIER_PATH, help="path to trained texture classifier pkl for mito post-processing")
    parser.add_argument("--partial-label-eval", action="store_true", default=False, help="evaluate only predicted components that overlap same-class GT")
    parser.add_argument("--partial-label-classes", type=str, default=None, help="comma-separated remapped class ids for partial-label evaluation; defaults to quantify classes")
    parser.add_argument("--partial-label-overlap-min-pixels", type=int, default=1, help="minimum same-class GT overlap pixels required to evaluate a predicted component")
    parser.add_argument("--visualize", action="store_true", default=False, help="save sample prediction figures")
    parser.add_argument("--num-vis-samples", type=int, default=10, help="number of samples to visualize")
    parser.add_argument("--no-progress", action="store_true", default=False, help="disable tqdm progress bars")
    return parser


def configure_args(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.show_progress = sys.stdout.isatty() and not args.no_progress
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        print("Using GPU ids: {}".format(args.gpu_ids))
    else:
        args.gpu_ids = []
        print("CUDA is disabled or unavailable. Evaluation will run on CPU.")

    if args.resume is None:
        args.resume = auto_resume_path(args.dataset)
    if args.resume is None:
        raise RuntimeError("No checkpoint was provided and no model_best.pth.tar was found under run/{}.".format(args.dataset))
    args.partial_label_class_ids = parse_int_list(args.partial_label_classes)


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_json_config_overrides(parser, args)
    args = finalize_jijie_args(args)
    configure_args(args)

    print(args)
    for line in summarize_jijie_run(args):
        print(line)

    evaluator = SegmentationEvaluator(args)
    summary = evaluator.run()

    print("Evaluation finished.")
    print("Pixel Accuracy: {:.4f}".format(summary["pixel_accuracy"]))
    print("Class Accuracy: {:.4f}".format(summary["pixel_accuracy_class"]))
    print("mIoU: {:.4f}".format(summary["mean_iou"]))
    print("Positive-class IoU: {:.4f}".format(summary["mean_positive_iou"]))
    print("Positive-class Dice: {:.4f}".format(summary["mean_positive_dice"]))
    partial_summary = summary.get("partial_label_overlap_summary") or {}
    if partial_summary:
        print("Partial-label overlap IoU: {:.4f}".format(partial_summary.get("mean_iou", 0.0)))
        print("Partial-label overlap Dice: {:.4f}".format(partial_summary.get("mean_dice", 0.0)))
    print("Artifacts saved to: {}".format(args.save_dir))


if __name__ == "__main__":
    main()
