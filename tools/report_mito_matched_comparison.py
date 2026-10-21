"""Compare paired five-fold outputs, with pooled and fold-macro metrics."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def load(root):
    result = {}
    for p in root.glob('*fold_*/metrics_summary.json'):
        fold = p.parent.name.rsplit('_', 1)[-1]
        if fold in result:
            raise ValueError('Duplicate fold: ' + fold)
        data = json.loads(p.read_text(encoding='utf-8'))
        with (p.parent / 'confusion_matrix.csv').open() as f:
            matrix = [[float(x) for x in row] for row in csv.reader(f)]
        assert len(matrix) == 5 and all(len(row) == 5 for row in matrix)
        result[fold] = (data, matrix)
    assert set(result) == {'00', '01', '02', '03', '04'}
    return result


def metrics(matrix, c):
    tp = matrix[c][c]
    gt = sum(matrix[c])
    pred = sum(row[c] for row in matrix)
    ratio = lambda a, b: a / b if b else 0.0
    return dict(dice=ratio(2 * tp, gt + pred), iou=ratio(tp, gt + pred - tp),
                precision=ratio(tp, pred), recall=ratio(tp, gt), gt_pixels=gt)


def write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--finetune', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    datasets = [load(args.baseline), load(args.finetune)]
    for fold in datasets[0]:
        a, b = [d[fold] for d in datasets]
        assert a[0]['class_names'] == b[0]['class_names']
        assert [sum(row) for row in a[1]] == [sum(row) for row in b[1]], 'GT support mismatch'
        for payload, matrix in (a, b):
            for c in range(1, 5):
                for key in ('dice', 'iou', 'precision', 'recall'):
                    assert abs(metrics(matrix, c)[key] - payload['per_class_metrics'][c][key]) < 1e-7
    pooled = [[[sum(d[f][1][i][j] for f in d) for j in range(5)] for i in range(5)] for d in datasets]
    class_rows = []
    for c in range(1, 5):
        values = [metrics(m, c) for m in pooled]
        row = dict(class_name=datasets[0]['00'][0]['class_names'][c], gt_pixels=int(values[0]['gt_pixels']))
        for key in ('dice', 'iou', 'precision', 'recall'):
            row['before_' + key] = values[0][key]
            row['after_' + key] = values[1][key]
            row['delta_pp_' + key] = 100 * (values[1][key] - values[0][key])
        class_rows.append(row)
    overall = []
    for key in ('dice', 'iou'):
        for method in ('fold_macro', 'pooled_class_macro'):
            vals = []
            for d, m in zip(datasets, pooled):
                v = ([d[f][0]['mean_positive_' + key] for f in sorted(d)] if method == 'fold_macro'
                     else [metrics(m, c)[key] for c in range(1, 5)])
                vals.append((statistics.mean(v), statistics.pstdev(v) if method == 'fold_macro' else None))
            overall.append(dict(aggregation=method, metric=key, before=vals[0][0], after=vals[1][0],
                                delta_pp=100*(vals[1][0]-vals[0][0]), before_fold_std=vals[0][1], after_fold_std=vals[1][1]))
    folds = []
    for f in sorted(datasets[0]):
        row = dict(fold=f)
        for key in ('dice', 'iou'):
            vals = [d[f][0]['mean_positive_' + key] for d in datasets]
            row['before_' + key], row['after_' + key] = vals
            row['delta_pp_' + key] = 100*(vals[1]-vals[0])
        folds.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'overall.csv', overall)
    write(args.output / 'per_class_pooled.csv', class_rows)
    write(args.output / 'per_fold.csv', folds)
    report = dict(overall=overall, per_class_pooled=class_rows, per_fold=folds,
                  caveats=['Ignored pixels excluded; not a full-image false-positive assessment.',
                           'Held-out folds also used for checkpoint selection; not independent testing.',
                           'Historical checkpoint training overlap has not been audited.',
                           'Revised annotations include retained AI objects.'])
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
