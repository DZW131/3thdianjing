"""Classify predicted MitoNet instances with the existing five-class DeepLab model."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mito_instance_features import pool_and_measure


def load_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def save_csv(path, rows):
    if not rows:
        raise ValueError("No instance classification rows were produced")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "checkpoint", "images", "mitonet", "samples-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")

    provenance = json.loads((args.mitonet / "provenance.json").read_text())
    if not provenance.get("complete"):
        raise ValueError("MitoNet instance run is incomplete")
    manifest = load_manifest(args.samples_csv)
    sample_ids = sorted(provenance["images"])
    if set(sample_ids) != set(manifest):
        raise ValueError("MitoNet output and source manifest sample IDs differ")

    sys.path.insert(0, str(args.project))
    from utils.jijie.inference import predict_logits
    from modeling.backbone import resnet
    resnet.ResNet._load_pretrained_model = lambda self: None
    from modeling.deeplab import DeepLab

    torch.set_num_threads(4)
    model = DeepLab(backbone="resnet", output_stride=16, num_classes=5, sync_bn=False).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)

    args.output.mkdir(parents=True)
    rows = []
    image_summaries = {}
    with torch.inference_mode():
        for sample_id in sample_ids:
            image_path = args.images / f"{sample_id}.jpg"
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if image_hash != provenance["images"][sample_id]["sha256"]:
                raise ValueError(f"Input image hash mismatch: {sample_id}")
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            normalized = (rgb.astype(np.float32) / 255.0 -
                          np.array([.485, .456, .406], dtype=np.float32)) / np.array(
                              [.229, .224, .225], dtype=np.float32)
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None].cuda()
            logits = predict_logits(model, tensor, 5, inference_mode="sliding",
                                     window_size=640, stride=320)
            probabilities = logits.softmax(1)[0].cpu().numpy()
            instance_path = args.mitonet / f"{sample_id}_instances.npz"
            labels = np.load(instance_path)["labels"]
            if probabilities.shape[1:] != labels.shape:
                raise ValueError(f"Image/instance shape mismatch: {sample_id}")

            object_rows = pool_and_measure(probabilities, labels)
            source = manifest[sample_id]
            class_map = np.zeros(labels.shape, dtype=np.uint8)
            for row in object_rows:
                row.update(sample_id=sample_id, case_id=source["case_id"],
                           slice_id=source["slice_id"],
                           source_tif=source["source_tif"], source_db=source["source_db"])
                class_map[labels == row["instance_id"]] = row["review_gate"]
                rows.append(row)
            Image.fromarray(class_map).save(args.output / f"{sample_id}_class_review_gate.png")
            counts = {str(class_id): int(sum(r["review_gate"] == class_id for r in object_rows))
                      for class_id in range(5)}
            image_summaries[sample_id] = dict(
                source_tif=source["source_tif"], source_db=source["source_db"],
                instances=int(labels.max()), classified_instances=int(sum(r["review_gate"] > 0 for r in object_rows)),
                needs_review=int(sum(r["needs_review"] for r in object_rows)), review_gate_counts=counts,
                image_sha256=image_hash)
            print(sample_id, image_summaries[sample_id], flush=True)
            del tensor, logits, probabilities, labels

    save_csv(args.output / "instances_with_class.csv", rows)
    class_names = {0: "unresolved_needs_review", 1: "severe", 2: "moderate",
                   3: "healthy", 4: "autophagic"}
    summary = dict(
        dataset="unlabeled cardiac TEM batch; not an evaluation set",
        image_count=len(sample_ids), objects=len(rows), class_names=class_names,
        classifier="unchanged five-class DeepLabV3+ baseline, sliding window 640/320",
        instance_source="MitoNet_v1 zero-shot panoptic predictions; instance IDs preserved",
        default_class_map="review_gate: class 0 means uncertain/review, not background; objects remain in MitoNet instance map",
        additional_per_object_rules={
            "mean_prob_all": "argmax mean 5-class probability including background",
            "mito_conditioned": "argmax mean 4 foreground probabilities; forces a damage class",
            "hard_vote": "majority pixel argmax including background"},
        review_gate="mitochondrial foreground support >= 0.5 and conditional top-two margin >= 0.1; heuristic, not calibrated confidence",
        damage_classes={1: "severe", 2: "moderate", 3: "healthy", 4: "autophagic"},
        units="pixels; no scale calibration applied", ids_preserved=True,
        caveats=["Predictions are for review, not clinical use.",
                 "No ground truth is available for this batch; metrics are not reported.",
                 "Class 0 in class_review_gate PNG means uncertain object pixels, while background is 0 in the separate instance-ID NPZ.",
                 "The DeepLab classifier was trained on earlier data and may be domain-shifted on this batch."],
        images=image_summaries)
    (args.output / "provenance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
