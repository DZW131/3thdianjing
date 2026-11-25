"""Package the two editable MitoNet slice.db pre-annotation variants."""
import argparse
import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


README = """MitoNet cardiac TEM editable pre-annotations

This archive contains copied/extended slice.db files only. Original TIFF files are not duplicated.
Match each database to the source TIFF using manifest.csv and samples.csv (source_tif / sample_id).
The original downloaded dataset and its databases were not modified.

01_instance_only: each predicted mitochondrion is an editable AI polygon in one unclassified group.
02_instance_plus_class: same MitoNet contours, grouped by DeepLab class candidate; uncertain objects are explicitly marked for review in each remark.

Predicted class IDs: 1 severe, 2 moderate, 3 healthy, 4 autophagic.
The classified version uses forced foreground probabilities to provide one candidate class per predicted instance; this is not calibrated confidence or an evaluation result.
Polygons are simplified (epsilon 1.5 px, at most 256 vertices) and objects below 8 pixels are omitted.
Disconnected regions within one panoptic ID are exported as separate editable polygons.
"""


def add_tree(archive, root, prefix):
    for path in sorted((root / "slice_files").rglob("slice.db")):
        archive.write(path, f"{prefix}/slice_files/{path.relative_to(root / 'slice_files').as_posix()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be fresh")
    args.output.mkdir(parents=True)
    specs = (
        ("instances", "mitonet_20261119_editable_instance_only.zip", "01_instance_only"),
        ("classified", "mitonet_20261119_editable_instance_plus_class.zip", "02_instance_plus_class"),
    )
    for mode, filename, prefix in specs:
        root = args.exports / mode
        target = args.output / filename
        with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            archive.write(root / "manifest.csv", f"{prefix}/manifest.csv")
            archive.write(root / "summary.json", f"{prefix}/summary.json")
            archive.write(args.samples, f"{prefix}/samples.csv")
            archive.writestr(f"{prefix}/README.txt", README)
            if mode == "classified":
                archive.write(args.classification_csv,
                              f"{prefix}/instances_with_class.csv")
            add_tree(archive, root, prefix)
        print(filename, target.stat().st_size)


if __name__ == "__main__":
    main()
