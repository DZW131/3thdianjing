import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap
from modeling.deeplab import DeepLab
from utils.checkpoint import load_checkpoint
from utils.metrics import Evaluator


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


def auto_resume_path(dataset_name):
    run_root = Path("run") / dataset_name
    if not run_root.is_dir():
        return None

    candidates = sorted(run_root.glob("*/model_best.pth.tar"))
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


def colorize_mask(mask, dataset_name):
    colored = decode_segmap(mask.astype(np.int64), dataset=dataset_name)
    return (colored * 255).astype(np.uint8)


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
        self.best_pred = checkpoint.get("best_pred", 0.0)
        if args.cuda:
            self.model = self.model.cuda()
        self.model.eval()

        self.evaluator = Evaluator(self.nclass)

    def run(self):
        self.evaluator.reset()
        vis_samples = []

        tbar = tqdm(self.loader, desc="Evaluating")
        with torch.no_grad():
            for sample in tbar:
                image = sample["image"]
                target = sample["label"]
                if self.args.cuda:
                    image = image.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = self.model(image)
                pred = torch.argmax(output, dim=1)

                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                self.evaluator.add_batch(target_np, pred_np)

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
                            }
                        )

        summary = self._build_summary()
        self._save_metrics(summary)
        self._save_confusion_matrix()
        if self.args.visualize:
            self._save_visualizations(vis_samples)
        return summary

    def _build_summary(self):
        confusion = self.evaluator.confusion_matrix
        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        per_class_rows = []
        for class_id in range(self.nclass):
            intersection = confusion[class_id, class_id]
            gt_total = confusion[class_id, :].sum()
            pred_total = confusion[:, class_id].sum()
            union = gt_total + pred_total - intersection

            iou = intersection / union if union > 0 else 0.0
            precision = intersection / pred_total if pred_total > 0 else 0.0
            recall = intersection / gt_total if gt_total > 0 else 0.0
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}",
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "pixel_count": int(gt_total),
                }
            )

        return {
            "dataset": self.args.dataset,
            "split": self.args.split,
            "resume": self.args.resume,
            "best_mIoU_from_checkpoint": float(self.best_pred),
            "pixel_accuracy": float(acc),
            "pixel_accuracy_class": float(acc_class),
            "mean_iou": float(miou),
            "frequency_weighted_iou": float(fwiou),
            "num_classes": int(self.nclass),
            "class_names": self.class_names,
            "per_class_metrics": per_class_rows,
        }

    def _save_metrics(self, summary):
        write_json(os.path.join(self.args.save_dir, "metrics_summary.json"), summary)
        write_csv(
            os.path.join(self.args.save_dir, "per_class_metrics.csv"),
            summary["per_class_metrics"],
            ["class_id", "class_name", "iou", "precision", "recall", "pixel_count"],
        )
        np.savetxt(
            os.path.join(self.args.save_dir, "confusion_matrix.csv"),
            self.evaluator.confusion_matrix,
            delimiter=",",
            fmt="%.0f",
        )

    def _save_confusion_matrix(self):
        confusion = self.evaluator.confusion_matrix.astype(np.float64)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names[: self.nclass],
            yticklabels=self.class_names[: self.nclass],
        )
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _save_visualizations(self, vis_samples):
        vis_dir = ensure_dir(os.path.join(self.args.save_dir, "visualizations"))
        for sample_index, sample in enumerate(vis_samples):
            image = denormalize_image(sample["image"])
            gt_mask = sample["target"].numpy().astype(np.uint8)
            pred_mask = sample["pred"].numpy().astype(np.uint8)

            gt_colored = colorize_mask(gt_mask, self.args.dataset)
            pred_colored = colorize_mask(pred_mask, self.args.dataset)
            overlay_gt = np.clip(0.6 * image + 0.4 * gt_colored, 0, 255).astype(np.uint8)
            overlay_pred = np.clip(0.6 * image + 0.4 * pred_colored, 0, 255).astype(np.uint8)
            diff_mask = np.where(gt_mask == pred_mask, 0, 255).astype(np.uint8)
            diff_colored = np.stack([diff_mask, np.zeros_like(diff_mask), np.zeros_like(diff_mask)], axis=2)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            panels = [
                (image, "Original"),
                (gt_colored, "Ground Truth"),
                (pred_colored, "Prediction"),
                (overlay_gt, "Original + GT"),
                (overlay_pred, "Original + Prediction"),
                (diff_colored, "Prediction Error"),
            ]
            for axis, (panel, title) in zip(axes.flat, panels):
                axis.imshow(panel)
                axis.set_title(title)
                axis.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{sample_index:03d}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Evaluation")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"])
    parser.add_argument("--out-stride", type=int, default=16, help="network output stride")
    parser.add_argument("--dataset", type=str, default="jijie", help="dataset name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="dataset split")
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
    parser.add_argument("--sync-bn", type=bool, default=False, help="kept for CLI compatibility")
    parser.add_argument("--freeze-bn", type=bool, default=False, help="freeze batch norm parameters")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N", help="training batch size placeholder")
    parser.add_argument("--test-batch-size", type=int, default=1, metavar="N", help="evaluation batch size")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA")
    parser.add_argument("--gpu-ids", type=str, default="0", help="comma-separated GPU ids")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--save-dir", type=str, default="outputs/eval", help="directory for metrics and figures")
    parser.add_argument("--visualize", action="store_true", default=False, help="save sample prediction figures")
    parser.add_argument("--num-vis-samples", type=int, default=10, help="number of samples to visualize")
    return parser


def configure_args(args):
    args.selected_classes = parse_selected_classes(args.selected_classes)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

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


def main():
    parser = build_parser()
    args = parser.parse_args()
    configure_args(args)

    print(args)
    evaluator = SegmentationEvaluator(args)
    summary = evaluator.run()

    print("Evaluation finished.")
    print("Pixel Accuracy: {:.4f}".format(summary["pixel_accuracy"]))
    print("Class Accuracy: {:.4f}".format(summary["pixel_accuracy_class"]))
    print("mIoU: {:.4f}".format(summary["mean_iou"]))
    print("FWIoU: {:.4f}".format(summary["frequency_weighted_iou"]))
    print("Artifacts saved to: {}".format(args.save_dir))


if __name__ == "__main__":
    main()
