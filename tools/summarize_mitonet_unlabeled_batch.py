"""Summarize full-batch MitoNet inference and object-level classifier outputs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = read_csv(args.samples)
    classified = read_csv(args.classification / "instances_with_class.csv")
    instance_meta = json.loads((args.instances / "provenance.json").read_text())
    class_meta = json.loads((args.classification / "provenance.json").read_text())
    rules = ("hard_vote", "mean_prob_all", "mito_conditioned", "review_gate")
    by_rule = {rule: Counter(int(row[rule]) for row in classified) for rule in rules}
    classes = {0: "unresolved/background", 1: "severe", 2: "moderate",
               3: "healthy", 4: "autophagic"}
    summary = dict(
        dataset="Fuwai cardiac TEM batch downloaded 2026-11-19",
        selected_samples=len(samples),
        sample_ids_with_instance_predictions=len(instance_meta["images"]),
        instance_predictions_complete=bool(instance_meta.get("complete")),
        object_rows=len(classified),
        predicted_instance_total=sum(int(row["instances"]) for row in instance_meta["images"].values()),
        classified_objects=sum(int(int(row["review_gate"]) > 0) for row in classified),
        objects_flagged_for_review=sum(str(row["needs_review"]).lower() == "true" for row in classified),
        class_names=classes,
        object_decision_counts={rule: {classes[key]: value for key, value in sorted(counts.items())}
                                for rule, counts in by_rule.items()},
        class_map_files=len(list((args.classification / "class_maps").glob("*.png"))),
        no_ground_truth_metrics="Not computed: this is an unlabeled batch.",
        model_details=dict(
            instance_model="official MitoNet_v1 TorchScript, zero-shot, full resolution",
            classifier="existing five-class DeepLabV3+ baseline, 640/320 sliding windows",
            instance_ids_preserved=True,
            uncertain_rule=class_meta["review_gate"]),
        source_mapping="samples.csv maps each generated image and prediction to the original relative TIFF and slice.db paths.",
        excluded_from_inference=json.loads((args.samples.parent / "inventory.json").read_text()).get("excluded", []),
        caveats=class_meta.get("caveats", []))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README.txt").write_text(
        "Fuwai TEM MitoNet pre-annotation batch\n\n"
        "1_instance_only: per-slice NPZ files with a full-resolution integer instance-ID map; 0 is background.\n"
        "2_instance_plus_class: the same instance IDs, per-instance CSV decisions, and three categorical PNG maps.\n"
        "Class maps: mean_prob_all (allows background), mito_conditioned (forces a foreground class), review_gate (uncertain object pixels are 0).\n"
        "Class IDs: 1 severe, 2 moderate, 3 healthy, 4 autophagic. Review-gate 0 means uncertain, not confirmed background.\n"
        "Use instances_with_class.csv and samples.csv to map objects to source TIFF/slice.db.\n"
        "MitoNet inference is zero-shot; classification uses the existing DeepLab baseline. The batch has no GT, so no accuracy metrics are claimed.\n"
        "The output is a research pre-annotation package, not a clinical result.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
