"""Run case-grouped five-fold fine-tuning and held-out mito evaluation.

Every fold starts from the same historic baseline checkpoint.  The supplied
train/validation manifests fully define the data used by a fold, so the runner
can evaluate either the 52-slice revised-only protocol or a larger,
provenance-weighted doctor-supervision dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", type=Path, default=root / "configs" / "mito" / "mito_v0_9_doctor_revision_cv5.json")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="isolated 52-slice prepared dataset root")
    parser.add_argument("--folds-root", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, default=None, help="optional training-only mask directory")
    parser.add_argument("--eval-mask-dir", type=Path, default=None, help="optional validation/test-only mask directory")
    parser.add_argument("--train-supervision-weight-dir", type=Path, default=None, help="optional training-only supervision weight directory")
    parser.add_argument("--eval-supervision-weight-dir", type=Path, default=None, help="optional validation/test-only supervision weight directory")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--run-prefix", default="jijie_mito_v0_9_doctor_revision_cv5")
    return parser.parse_args()


def run_logged(command, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[Run] {}".format(" ".join(str(item) for item in command)))
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    config = args.config.resolve()
    checkpoint = args.baseline_checkpoint.resolve()
    data_root = args.data_root.resolve()
    folds_root = args.folds_root.resolve()
    split_dirs = {
        "--train-mask-dir": args.train_mask_dir,
        "--eval-mask-dir": args.eval_mask_dir,
        "--train-supervision-weight-dir": args.train_supervision_weight_dir,
        "--eval-supervision-weight-dir": args.eval_supervision_weight_dir,
    }
    split_dirs = {flag: directory.resolve() for flag, directory in split_dirs.items() if directory is not None}
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    if not (data_root / "JPEGImages").is_dir() or not (data_root / "SegmentationClass").is_dir():
        raise FileNotFoundError("invalid prepared data root: {}".format(data_root))

    logs_root = project_root / "logs" / args.run_prefix
    eval_root = project_root / "outputs" / "eval" / args.run_prefix
    run_metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_checkpoint": str(checkpoint),
        "config": str(config),
        "data_root": str(data_root),
        "folds_root": str(folds_root),
        "split_specific_label_directories": {flag: str(directory) for flag, directory in split_dirs.items()},
        "protocol": "five-fold case-grouped CV; each fold loads the same baseline then uses only its own train/validation manifests",
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for fold in range(args.fold_count):
        fold_name = "fold_{:02d}".format(fold)
        manifest_dir = folds_root / fold_name
        if not (manifest_dir / "train.txt").is_file() or not (manifest_dir / "val.txt").is_file():
            raise FileNotFoundError("missing manifests under {}".format(manifest_dir))
        checkname = "{}_{}".format(args.run_prefix, fold_name)
        train_command = [
            args.python,
            "train.py",
            "--config", str(config),
            "--data-root", str(data_root),
            "--manifest-dir", str(manifest_dir),
            "--resume", str(checkpoint),
            "--ft",
            "--checkname", checkname,
            "--workers", str(args.workers),
            "--gpu-ids", args.gpu_ids,
            "--no-progress",
        ]
        for flag, directory in split_dirs.items():
            train_command.extend([flag, str(directory)])
        run_logged(train_command, logs_root / "{}_train.log".format(fold_name), project_root)

        best_checkpoint = project_root / "run" / "jijie" / checkname / "model_best.pth.tar"
        if not best_checkpoint.is_file():
            raise RuntimeError("training did not produce best checkpoint: {}".format(best_checkpoint))
        evaluate_command = [
            args.python,
            "evaluate.py",
            "--config", str(config),
            "--data-root", str(data_root),
            "--manifest-dir", str(manifest_dir),
            "--split", "test",
            "--resume", str(best_checkpoint),
            "--save-dir", str(eval_root / checkname),
            "--workers", str(args.workers),
            "--gpu-ids", args.gpu_ids,
            "--partial-label-eval",
            "--no-progress",
        ]
        for flag, directory in split_dirs.items():
            evaluate_command.extend([flag, str(directory)])
        run_logged(evaluate_command, logs_root / "{}_evaluate.log".format(fold_name), project_root)

    summary_command = [
        args.python,
        "tools/summarize_mito_doctor_revision_cv.py",
        "--eval-root", str(eval_root),
        "--output-dir", str(eval_root / "summary"),
    ]
    run_logged(summary_command, logs_root / "summarize.log", project_root)
    print("[Done] cross-validation artifacts: {}".format(eval_root))


if __name__ == "__main__":
    main()
