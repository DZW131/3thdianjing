import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataloaders import custom_transforms as tr
from evaluate import build_parser, parse_int_list
from modeling.deeplab import DeepLab
from utils.checkpoint import load_checkpoint, state_dict_from_model
from utils.jijie import apply_json_config_overrides, finalize_jijie_args
from utils.jijie.inference import predict_logits
from utils.jijie.postprocess import DEFAULT_CLASSIFIER_PATH, postprocess_mito_mask
from utils.jijie.visualization import blend_mask, colorize_mask, palette_rows, to_display_class_names


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item:
                ids.append(item)
    return ids


def unique_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def collect_task_ids(manifest_dir, splits):
    ids = []
    for split in splits:
        split_file = manifest_dir / f"{split}.txt"
        if split_file.is_file():
            ids.extend(read_ids(split_file))
    return unique_preserve_order(ids)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_args(config_path, resume_path, save_dir):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--resume",
            str(resume_path),
            "--save-dir",
            str(save_dir),
            "--split",
            "test",
            "--no-progress",
        ]
    )
    args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")] if isinstance(args.gpu_ids, str) else args.gpu_ids
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args = apply_json_config_overrides(parser, args)
    args = finalize_jijie_args(args)
    return args


def build_transform(args):
    transforms_list = []
    resize_mode = getattr(args, "eval_resize_mode", getattr(args, "resize_mode", "none"))
    crop_size = getattr(args, "crop_size", 512)
    if resize_mode == "pad":
        transforms_list.append(tr.ResizeLongestSideAndPad(crop_size))
    elif resize_mode == "resize":
        transforms_list.append(tr.FixedResize(crop_size))
    elif resize_mode == "crop":
        transforms_list.append(tr.FixScaleCrop(crop_size))
    elif resize_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported eval resize mode for export: {resize_mode}")
    transforms_list.extend(
        [
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ]
    )
    return transforms.Compose(transforms_list)


def load_model(args, nclass):
    model = DeepLab(
        num_classes=nclass,
        backbone=args.backbone,
        output_stride=args.out_stride,
        sync_bn=args.sync_bn,
        freeze_bn=args.freeze_bn,
    )
    if args.cuda:
        model = model.cuda()
    checkpoint = load_checkpoint(args.resume, map_location="cuda" if args.cuda else "cpu")
    model.load_state_dict(state_dict_from_model(checkpoint["state_dict"]))
    model.eval()
    return model


def local_to_original_mask(local_mask, selected_classes):
    selected = [int(class_id) for class_id in selected_classes]
    out = np.zeros_like(local_mask, dtype=np.uint8)
    for local_id, original_id in enumerate(selected, start=1):
        out[local_mask == local_id] = original_id
    return out


def export_task(
    task_name,
    config_path,
    resume_path,
    data_root,
    output_root,
    splits,
    all_images,
    postprocess_mito=False,
    postprocess_classifier=None,
):
    save_dir = output_root / task_name
    review_dir = ensure_dir(save_dir / "review_dataset")
    image_out_dir = ensure_dir(review_dir / "JPEGImages")
    mask_out_dir = ensure_dir(review_dir / "SegmentationClass")
    color_dir = ensure_dir(save_dir / "color_masks")
    overlay_dir = ensure_dir(save_dir / "overlays")

    args = build_eval_args(config_path, resume_path, save_dir)
    selected_classes = parse_int_list(args.selected_classes)
    class_names = ["Background"] + list(getattr(args, "selected_class_names", [str(c) for c in selected_classes]))
    display_class_names = to_display_class_names(class_names, language="zh")
    nclass = len(class_names)

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = Path.cwd() / manifest_dir
    if all_images:
        ids = sorted(path.stem for path in (data_root / "JPEGImages").glob("*.jpg"))
    else:
        ids = collect_task_ids(manifest_dir, splits)
    if not ids:
        raise RuntimeError(f"No image ids found for task {task_name}")

    model = load_model(args, nclass)
    transform = build_transform(args)
    rows = []
    device = torch.device("cuda" if args.cuda else "cpu")

    with torch.no_grad():
        for index, sample_id in enumerate(ids, start=1):
            image_path = data_root / "JPEGImages" / f"{sample_id}.jpg"
            if not image_path.is_file():
                print(f"[Skip] missing image: {image_path}")
                continue
            image = Image.open(image_path).convert("RGB")
            sample = {
                "image": image,
                "label": Image.new("L", image.size, 0),
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": "",
                "original_size": image.size[::-1],
                "task_name": task_name,
            }
            tensor_sample = transform(sample)
            tensor = tensor_sample["image"].unsqueeze(0).to(device)
            logits = predict_logits(
                model,
                tensor,
                nclass,
                inference_mode=args.inference_mode,
                window_size=args.sliding_window_size,
                stride=args.sliding_window_stride,
            )
            pred_local = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            if pred_local.shape != (image.height, image.width):
                pred_local = np.array(
                    Image.fromarray(pred_local).resize((image.width, image.height), Image.NEAREST),
                    dtype=np.uint8,
                )
            if task_name == "mito" and postprocess_mito:
                gray = np.asarray(image.convert("L"))
                pred_local, _ = postprocess_mito_mask(
                    pred_local.astype(np.int32),
                    gray_image=gray,
                    classifier_path=str(postprocess_classifier) if postprocess_classifier else None,
                    use_texture_refine=bool(postprocess_classifier and Path(postprocess_classifier).is_file()),
                )
                pred_local = pred_local.astype(np.uint8)
            pred_original = local_to_original_mask(pred_local, selected_classes)

            image_copy_path = image_out_dir / f"{sample_id}.jpg"
            pred_mask_path = mask_out_dir / f"{sample_id}.png"
            color_mask_path = color_dir / f"{sample_id}.png"
            overlay_path = overlay_dir / f"{sample_id}.jpg"

            if not image_copy_path.exists():
                shutil.copy2(image_path, image_copy_path)
            Image.fromarray(pred_original).save(pred_mask_path)
            colored = colorize_mask(pred_local, class_names)
            Image.fromarray(colored).save(color_mask_path)
            overlay = blend_mask(np.asarray(image), colored, alpha=0.42)
            Image.fromarray(overlay).save(overlay_path, quality=95)

            rows.append(
                {
                    "sample_id": sample_id,
                    "image": str(image_copy_path.relative_to(save_dir)),
                    "prediction_mask": str(pred_mask_path.relative_to(save_dir)),
                    "color_mask": str(color_mask_path.relative_to(save_dir)),
                    "overlay": str(overlay_path.relative_to(save_dir)),
                }
            )
            if index % 20 == 0 or index == len(ids):
                print(f"[{task_name}] {index}/{len(ids)} exported")

    write_csv(save_dir / "manifest.csv", rows, ["sample_id", "image", "prediction_mask", "color_mask", "overlay"])
    write_csv(
        save_dir / "class_palette.csv",
        palette_rows(class_names, display_class_names),
        ["class_id", "class_name", "class_name_display", "color_r", "color_g", "color_b", "hex_color"],
    )
    summary = {
        "task_name": task_name,
        "config": str(config_path),
        "resume": str(resume_path),
        "num_exported": len(rows),
        "mask_values": {str(local_id): int(original_id) for local_id, original_id in enumerate([0] + selected_classes)},
        "postprocess_mito": bool(task_name == "mito" and postprocess_mito),
        "review_dataset": str(review_dir),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export full-image predictions for doctor review.")
    parser.add_argument("--data-root", type=Path, default=Path("data/jijie"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/doctor_review_predictions_mito_mito_sr"))
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--all-images", action="store_true", help="export every JPEGImages/*.jpg instead of task final_200 split ids")
    parser.add_argument("--mito-config", type=Path, default=Path("configs/mito/mito_v0_5_final200.json"))
    parser.add_argument("--mito-resume", type=Path, default=Path("run/jijie/jijie_mito_v0_5_final200/model_best.pth.tar"))
    parser.add_argument("--mito-sr-config", type=Path, default=Path("configs/mito_sr/mito_sr_v0_5_final200.json"))
    parser.add_argument("--mito-sr-resume", type=Path, default=Path("run/jijie/jijie_mito_sr_v0_5_final200/model_best.pth.tar"))
    parser.add_argument("--postprocess-mito", action="store_true", help="apply current mito instance/class post-processing before export")
    parser.add_argument("--postprocess-classifier", type=Path, default=Path(DEFAULT_CLASSIFIER_PATH))
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    summaries = []
    summaries.append(
        export_task(
            "mito",
            args.mito_config,
            args.mito_resume,
            args.data_root,
            output_root,
            splits,
            args.all_images,
            postprocess_mito=args.postprocess_mito,
            postprocess_classifier=args.postprocess_classifier,
        )
    )
    summaries.append(
        export_task("mito_sr", args.mito_sr_config, args.mito_sr_resume, args.data_root, output_root, splits, args.all_images)
    )
    (output_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Export finished:", output_root)


if __name__ == "__main__":
    main()
