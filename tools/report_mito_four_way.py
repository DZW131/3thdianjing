"""Summarize four matched model/postprocessing evaluations without extra dependencies."""
import argparse
import json
import statistics
from pathlib import Path

from report_mito_matched_comparison import load, metrics, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = {
        'baseline': 'jijie_mito_v0_5_currentraw_matched_20261014',
        'baseline_post': 'mito52_baseline_postprocess_matched_20261014',
        'finetune': 'jijie_mito_v0_13_revised52_currentraw_baseline_cv5',
        'finetune_post': 'mito52_finetune_postprocess_matched_20261014',
    }
    overall, classes, folds, audit = [], [], [], []
    reference = None
    for name, folder in names.items():
        data = load(args.raw / folder)
        support = {f: [sum(r) for r in m] for f, (_, m) in data.items()}
        if reference is None:
            reference = support
        assert support == reference, 'GT support mismatch'
        pooled = [[sum(data[f][1][i][j] for f in data) for j in range(5)] for i in range(5)]
        for f, (payload, matrix) in sorted(data.items()):
            if name.endswith('_post'):
                post = payload['postprocess']
                assert post['enabled'] and post['images_processed'] == {'00':17, '01':10, '02':10, '03':9, '04':6}[f]
                audit.append(dict(model=name, fold=f, **post))
            folds.append(dict(model=name, fold=f, dice=payload['mean_positive_dice'], iou=payload['mean_positive_iou']))
            for c in range(1, 5):
                for k in ('dice', 'iou'):
                    assert abs(metrics(matrix, c)[k] - payload['per_class_metrics'][c][k]) < 1e-7
        for k in ('dice', 'iou'):
            vals = [v[0]['mean_positive_' + k] for v in data.values()]
            overall.append(dict(model=name, metric=k, fold_mean=statistics.mean(vals), fold_std=statistics.pstdev(vals),
                                pooled_macro=statistics.mean(metrics(pooled,c)[k] for c in range(1,5))))
        for c in range(1,5):
            classes.append(dict(model=name, class_name=data['00'][0]['class_names'][c], **metrics(pooled,c)))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('four_way_overall.csv',overall), ('four_way_classes.csv',classes),
                           ('four_way_folds.csv',folds), ('postprocess_audit.csv',audit)]:
        write(args.output / filename, rows)
    result = dict(overall=overall, classes=classes, postprocess_audit=audit)
    (args.output / 'four_way.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
