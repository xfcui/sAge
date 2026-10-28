"""Evaluate saved feature masks and export one fold's segmented-elbow genes.

Used by elbow-gene-choose.ipynb. Regression uses the stored expression values,
as in the original notebook; it does not apply the classifier's normalization.
"""

import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits


AGE_MAPPING = {0: 1, 1: 3, 2: 18, 3: 21, 4: 24, 5: 30}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_genes(path):
    genes = Path(path).read_text(encoding='utf-8-sig').splitlines()
    genes = [gene.strip() for gene in genes]
    if not genes or any(not gene or len(gene.split()) != 1 for gene in genes):
        raise ValueError('The gene header must have one nonempty gene identifier per line.')
    # Never sort, uppercase, or deduplicate: each line represents one matrix column.
    return genes


def load_data(path, gene_count, age_mapping):
    with h5py.File(path, 'r') as handle:
        x = handle['data'][:]
        labels = handle['label'][:]
    if x.ndim != 2 or x.shape[1] != gene_count:
        raise ValueError(f'{path}: expression columns must match all {gene_count} header lines.')
    if labels.ndim != 2 or labels.shape[0] != len(x) or labels.shape[1] < 1:
        raise ValueError(f'{path}: invalid label shape {labels.shape}.')
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f'{path}: need at least two cells with finite expression values.')
    ids = labels[:, 0]
    if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError(f'{path}: age class IDs must be finite integers.')
    missing = set(ids.astype(int)) - set(age_mapping)
    if missing:
        raise ValueError(f'{path}: AGE_MAPPING is missing class IDs {sorted(missing)}.')
    ages = np.array([age_mapping[int(i)] for i in ids], dtype=float)
    if not np.isfinite(ages).all() or len(np.unique(ages)) < 2:
        raise ValueError(f'{path}: need at least two distinct finite ages.')
    return x, ages


def read_mask(path, gene_count):
    mask = np.loadtxt(path, ndmin=1)
    if mask.shape != (gene_count,) or not np.isin(mask, [0, 1]).all():
        raise ValueError(f'{path}: expected a binary mask with {gene_count} positions.')
    if not mask.any():
        raise ValueError(f'{path}: no selected features.')
    return mask.astype(bool)


def collect_masks(checkpoint_dir, gene_count, max_features):
    masks = []
    for folder in Path(checkpoint_dir).glob('feature*'):
        match = re.fullmatch(r'feature(\d+)', folder.name)
        if not folder.is_dir() or not match:
            continue
        count = int(match.group(1))
        if max_features is not None and count > max_features:
            continue
        path = folder / 'feature.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing exported feature mask: {path}')
        mask = read_mask(path, gene_count)
        if int(mask.sum()) != count:
            raise ValueError(f'{path}: mask sum disagrees with folder feature count {count}.')
        masks.append((count, path, mask))
    masks.sort(key=lambda item: item[0])
    if len(masks) < 6:
        raise ValueError(
            f'Found {len(masks)} candidate masks (max_features={max_features}); '
            'the elbow requires at least 6 distinct feature counts. '
            'Finish a full pruning run (--max_epochs 9999). '
            'The one-epoch Quick run and eight-epoch pruning check are insufficient. '
            'Also check RUN_DIR, FOLD and MAX_FEATURES.'
        )
    if len({count for count, _, _ in masks}) != len(masks):
        raise ValueError('Multiple checkpoint folders encode the same feature count.')
    return masks


def regression_metrics(x_train, y_train, x_eval, y_eval):
    # Bound BLAS threads to avoid oversubscription across many small regressions.
    with threadpool_limits(limits=1, user_api='blas'):
        regression = LinearRegression().fit(x_train, y_train)
        predicted = regression.predict(x_eval)
    pcc = float(pearsonr(y_eval, predicted).statistic) if np.ptp(predicted) > 0 else None
    return {'PCC': pcc, 'MAE': float(mean_absolute_error(y_eval, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(y_eval, predicted)))}


def segmented_knee(rows):
    """Original MAD-filtered, pre-minimum search, with corrected residual squares.

    Each side contains at least three points; the breakpoint is shared. Flat
    curves and insufficient evidence fail explicitly instead of picking a gene
    set arbitrarily. Ties choose the smaller feature count.
    """
    rows = sorted(rows, key=lambda row: row['Gene_Count'])
    x = np.array([row['Gene_Count'] for row in rows], dtype=float)
    y = np.array([row['MAE'] for row in rows], dtype=float)
    if len(x) < 6 or len(np.unique(x)) != len(x) or not np.isfinite(y).all():
        raise ValueError('Need at least six distinct feature counts with finite MAE.')
    if np.allclose(y, y[0], rtol=1e-10, atol=1e-12):
        raise ValueError('The MAE curve is flat; it does not identify an elbow.')
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    keep = np.abs(y - median) < 5 * mad if mad > 0 else np.ones(len(y), dtype=bool)
    retained = np.flatnonzero(keep)
    if len(retained) < 5:
        raise ValueError('Fewer than five points remain after MAD filtering; no defensible elbow.')
    xc, yc = x[keep], y[keep]
    end = max(4, int(np.argmin(yc)))
    xs, ys = xc[:end + 1], yc[:end + 1]
    candidates = []
    for i in range(2, len(xs) - 2):
        rss = 0.0
        for part in (slice(None, i + 1), slice(i, None)):
            xx, yy = xs[part].reshape(-1, 1), ys[part]
            residual = yy - LinearRegression().fit(xx, yy).predict(xx)
            rss += float(np.sum(residual ** 2))
        candidates.append((rss, i))
    rss, index = min(candidates)
    selected = dict(rows[int(retained[index])])
    selected['Segmented_RSS'] = rss
    selected['Knee_Search_Counts'] = xs.astype(int).tolist()
    selected['MAD_Excluded_Counts'] = x[~keep].astype(int).tolist()
    return selected


def plot_metrics(rows, selected, title, split_label):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    search = set(selected['Knee_Search_Counts'])
    search_rows = [row for row in rows if row['Gene_Count'] in search]
    for axes_row, plotted, caption in zip(axes, (rows, search_rows),
                                         ('All evaluated counts', 'Elbow search after MAD filtering')):
        counts = [row['Gene_Count'] for row in plotted]
        for axis, metric, units in zip(axes_row, ('PCC', 'MAE', 'RMSE'),
                                      ('Pearson correlation', 'MAE (months)', 'RMSE (months)')):
            values = [np.nan if row[metric] is None else row[metric] for row in plotted]
            axis.plot(counts, values, '.-', color='#24689b', linewidth=1, markersize=3)
            axis.axvline(selected['Gene_Count'], color='#333333', linestyle='--', linewidth=1,
                         label=f"Selected: {selected['Gene_Count']} genes")
            if selected[metric] is not None:
                axis.scatter(selected['Gene_Count'], selected[metric], marker='*', s=90,
                             color='#b56816', zorder=3)
            axis.set(xlabel='Number of retained genes', ylabel=units, title=caption)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=9)
    figure.suptitle(f'{title} | segmented MAE elbow\n{split_label}')
    return figure


def run_selection(data_dir, run_dir, header_file, fold, output_dir, *,
                  selection_split, max_features=1011, age_mapping=None):
    """Run on exactly one fold; selection_split must be chosen explicitly."""
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    header_file, output_dir = Path(header_file).resolve(), Path(output_dir).resolve()
    if not re.fullmatch(r'fold_\d+', fold):
        raise ValueError('FOLD must be a fold name such as fold_0.')
    if max_features is not None and (not isinstance(max_features, int) or max_features < 1):
        raise ValueError('MAX_FEATURES must be a positive integer or None.')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Use a new OUTPUT_DIR to preserve previous results: {output_dir}')
    fold_dir = run_dir / fold
    log = fold_dir / 'train.log'
    if not log.is_file():
        raise FileNotFoundError(f'Missing training log: {log}. Use the README training runner.')
    log_text = log.read_text(encoding='utf-8', errors='replace')
    if '#done!!!' not in log_text or '#Final Test:' not in log_text:
        raise ValueError(f'{fold}: training has not completed successfully. Check {log}.')
    if '#Early stopping triggered' not in log_text:
        raise ValueError(f'{fold}: no natural early stop recorded. Finish full pruning with '
                         '--max_epochs 9999 in a new run directory before final gene selection.')
    if selection_split == 'validation':
        train_path = data_dir / 'cv_folds' / fold / 'train.h5'
        evaluation_path = data_dir / 'cv_folds' / fold / 'valid.h5'
        split_label = 'CV validation (used for selection)'
    elif selection_split == 'initial_test':
        train_path = data_dir / 'initial_split' / 'train.h5'
        evaluation_path = data_dir / 'initial_split' / 'test.h5'
        split_label = 'Initial test (used for selection; not independent test performance)'
        print('NOTE: initial_split/test.h5 participates in gene selection. '
              'These metrics are not independent final-test estimates.')
    else:
        raise ValueError("SELECTION_SPLIT must be 'validation' or 'initial_test'.")
    genes = load_genes(header_file)
    mapping = AGE_MAPPING if age_mapping is None else age_mapping
    masks = collect_masks(fold_dir / 'checkpoints', len(genes), max_features)
    x_train, y_train = load_data(train_path, len(genes), mapping)
    x_eval, y_eval = load_data(evaluation_path, len(genes), mapping)
    print(f'{data_dir.name}, {fold}: {len(genes)} genes, {len(masks)} masks; '
          f'{len(y_train)} training / {len(y_eval)} selection cells.')
    rows = []
    for number, (count, path, mask) in enumerate(masks, 1):
        row = {'Gene_Count': count, 'Mask_Path': str(path)}
        row.update(regression_metrics(x_train[:, mask], y_train, x_eval[:, mask], y_eval))
        rows.append(row)
        if number == 1 or number % 10 == 0 or number == len(masks):
            print(f'  Evaluated {number}/{len(masks)} masks; {count} genes, MAE={row["MAE"]:.4f}')
    selected = segmented_knee(rows)
    mask = read_mask(selected['Mask_Path'], len(genes))
    indices = np.flatnonzero(mask)
    selected_genes = [genes[i] for i in indices]
    figure = plot_metrics(rows, selected, f'{data_dir.name} / {fold}', split_label)
    manifest = {
        'dataset': data_dir.name, 'fold': fold, 'selection_split': selection_split,
        'selection_metrics_are_independent_test_metrics': False,
        'max_features': max_features, 'age_mapping_months': mapping,
        'regression_input': 'stored HDF5 expression; no additional normalization',
        'method': 'MAD-filtered segmented MAE elbow; sum of squared residuals',
        'implementation_sha256': sha256(__file__),
        'package_versions': {package: version(package) for package in
                             ('numpy', 'scipy', 'scikit-learn', 'h5py', 'matplotlib')},
        'selected': selected,
        'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in
                   [('header', header_file), ('train', train_path), ('selection', evaluation_path),
                    ('training_log', log)]},
        'candidate_masks': [{'path': str(path), 'sha256': sha256(path)} for _, path, _ in masks],
    }
    command_path = fold_dir / 'command.json'
    if command_path.is_file():
        manifest['training_command'] = json.loads(command_path.read_text(encoding='utf-8'))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'feature_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / 'final_aging_genes.txt').write_text('\n'.join(selected_genes) + '\n', encoding='utf-8')
    with (output_dir / 'final_aging_genes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['feature_index_0based', 'gene'])
        writer.writerows(zip(indices.tolist(), selected_genes))
    np.savetxt(output_dir / 'selected_feature_mask.txt', mask.astype(int), fmt='%d')
    (output_dir / 'selection_summary.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    figure.savefig(output_dir / 'elbow_metrics.png', dpi=160)
    figure.savefig(output_dir / 'elbow_metrics.pdf')
    print(f'Selected {len(indices)} genes from {fold}. Results: {output_dir}')
    return rows, manifest, figure
