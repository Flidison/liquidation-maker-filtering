"""Descriptive probability calibration using saved development/test predictions.

No estimator or probability calibrator is fitted; no predict_proba is called.
Bins are defined by validation probability quantiles, with ties kept together.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import evt_log_cond_modeling as original
import evt_log_cond_reduced_5s as reduced
import evt_log_cond_test_reduced_5s as assessment

OUT = assessment.OUT / 'calibration'
MODELS = [reduced.MODEL_NAME, 'lightgbm_original', 'logistic_original']
LABELS = ['LightGBM · 5 деревьев', 'LightGBM · 15 деревьев', 'Логистическая регрессия']
COLORS = ['#3568A8', '#B88616', '#A54B78']


def bin_boundaries(validation_probabilities):
    p = np.asarray(validation_probabilities)
    cuts = np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1]))
    cuts = cuts[(cuts > p.min()) & (cuts < p.max())]
    return np.r_[0., cuts, 1.]


def grouped_calibration(y, p, edges):
    y, p = np.asarray(y), np.asarray(p)
    assert len(y) == len(p) and len(y) > 0
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    bins = np.searchsorted(edges[1:-1], p, side='right')
    f = pd.DataFrame({'bin': bins, 'target': y, 'probability': p})
    rows = f.groupby('bin').agg(n=('target', 'size'), positives=('target', 'sum'),
                               predicted=('probability', 'mean'), observed=('target', 'mean')).reset_index()
    rows['lower'] = edges[rows.bin]
    rows['upper'] = edges[rows.bin + 1]
    rows['absolute_gap'] = (rows.predicted - rows.observed).abs()
    assert rows.n.sum() == len(y) and rows.positives.sum() == y.sum()
    return rows


def summarize():
    assessment.verify_results()
    development = pd.read_parquet(reduced.OUT / 'predictions_development.parquet')
    validation = development.loc[development['split'].eq('validation')]
    test = pd.read_parquet(assessment.OUT / 'predictions_test.parquet')
    tables, scores, boundaries = [], [], {}
    for name in MODELS:
        edges = bin_boundaries(validation[name])
        boundaries[name] = edges.tolist()
        for split, frame in [('validation', validation), ('test', test)]:
            table = grouped_calibration(frame.target, frame[name], edges)
            table['model'], table['split'] = name, split
            tables.append(table)
            p, y = frame[name].to_numpy(), frame.target.to_numpy()
            scores.append({'model': name, 'split': split, 'n': len(frame),
                           'positives': int(y.sum()), 'mean_probability': float(p.mean()),
                           'observed_frequency': float(y.mean()),
                           'bias': float(p.mean()-y.mean()),
                           'ece': float(np.sum(table.n * table.absolute_gap) / len(frame)),
                           'brier_mse': float(np.mean((p-y)**2)),
                           'nonempty_bins': len(table)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(OUT / 'calibration_bins.csv', index=False)
    pd.DataFrame(scores).to_csv(OUT / 'calibration_metrics.csv', index=False)
    original.dump(OUT / 'method.json', {
        'quantiles': 10, 'edges_from': 'validation predicted probabilities only',
        'identical_probabilities_kept_together': True, 'edges': boundaries,
        'prediction_source_sha256': original.digest(assessment.OUT / 'predictions_test.parquet'),
        'development_source_sha256': original.digest(reduced.OUT / 'predictions_development.parquet'),
        'calibrator_fitted': False, 'new_prediction_calls': 0,
        'interpretation': 'Descriptive binned calibration; ECE depends on bins, '
                          'is affected by class imbalance and is not a probability of correctness.'})
    return pd.DataFrame(scores)


def calibration_figure():
    table = pd.read_csv(OUT / 'calibration_bins.csv')
    test = table.loc[table['split'].eq('test')]
    upper = max(test.predicted.max(), test.observed.max()) * 1.12
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True, sharey=True)
    for ax, name, label, color in zip(axes, MODELS, LABELS, COLORS):
        rows = test.loc[test.model.eq(name)].sort_values('predicted')
        ax.plot([0, upper], [0, upper], '--', color='#555555', linewidth=1.2, label='Точная калибровка')
        ax.plot(rows.predicted, rows.observed, 'o-', color=color, linewidth=1.8,
                markersize=5, label='Наблюдаемая частота')
        ax.set(xlim=(0, upper), ylim=(0, upper), title=label,
               xlabel='Средняя прогнозируемая вероятность')
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('Фактическая частота burst в следующие 5 с')
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Калибровка вероятностей · BTCUSDT + ETHUSDT · test 14–28 февраля 2026 UTC', fontsize=13)
    fig.text(.5, .035, '160 901 DC-наблюдение; 631 положительный target. Границы групп: квантили validation, равные прогнозы не разделяются.\n'
             'Точки — оценки частоты по группам, без доверительных интервалов; шкалы одинаковы. Использованы сохранённые предсказания.',
             ha='center', va='bottom', fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.22, wspace=.17)
    return fig


if __name__ == '__main__':
    print(summarize().to_string(index=False))
    fig = calibration_figure()
    fig.savefig(OUT / 'calibration_5s_test.png', dpi=160)
    plt.close(fig)
