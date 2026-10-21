"""Run the selected baseline sarcomere model on full EM images for DB export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_doctor_review_predictions import ensure_dir, export_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full-image baseline sarcomere predictions.")
    parser.add_argument("--data-root", type=Path, default=Path("data/jijie"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/doctor_review_predictions_sarcomere_baseline"),
    )
    parser.add_argument("--all-images", action="store_true", help="Predict every JPEGImages/*.jpg image.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sarcomere/sarcomere_v0_5_final200.json"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("run/jijie/jijie_sarcomere_v0_5_final200_5090_retrain/model_best.pth.tar"),
    )
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    summary = export_task(
        "sarcomere",
        args.config,
        args.resume,
        args.data_root,
        output_root,
        splits=["train", "val", "test"],
        all_images=args.all_images,
        max_samples=args.max_samples,
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
