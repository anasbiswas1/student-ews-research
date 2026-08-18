"""NB07 tooling, extracted verbatim (defs + constants only) for NB08.
Source: notebooks/07_scientific_validity_repairs.ipynb @ b471d93."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42


RNG = np.random.default_rng(SEED)


CUTOFF_WEEKS = [5, 10, 15, 25]


THRESHOLD = 0.50


KEYS = ['code_module', 'code_presentation', 'id_student']


PROTECTED = ['imd_band', 'disability', 'age_band', 'gender']


CONTEXT = ['region', 'highest_education']


try:
    from google.colab import drive  # noqa
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/StudentEWS_Research/student-ews-research')
except Exception:
    ROOT = Path('.')

DATA = ROOT / 'data'


ORIGINAL_PROC = ROOT / 'results' / 'processed'


OUT = ROOT / 'results' / 'scientific_revision'


PROC = OUT / 'processed'


MODELS = OUT / 'models'


def load_any(stem: Path) -> pd.DataFrame:
    for suffix, reader in [('.parquet', pd.read_parquet), ('.csv', pd.read_csv)]:
        path = stem.with_suffix(suffix)
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f'No .parquet or .csv file found for {stem}')


def save_any(df: pd.DataFrame, stem: Path) -> Path:
    try:
        path = stem.with_suffix('.parquet')
        df.to_parquet(path, index=False)
    except Exception:
        path = stem.with_suffix('.csv')
        df.to_csv(path, index=False)
    return path


def expected_calibration_error(y_true: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return float(sum(
        (bins == k).mean() * abs(y[bins == k].mean() - p[bins == k].mean())
        for k in range(n_bins) if np.any(bins == k)
    ))


def clean_imd_band(df: pd.DataFrame) -> pd.DataFrame:
    order = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
             '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    out = df.copy()
    s = out['imd_band'].astype('object').replace({'10-20': '10-20%'})
    s = s.where(s.isin(order), other='Unknown')
    out['imd_band_clean'] = pd.Categorical(s, categories=order + ['Unknown'], ordered=True)
    codes = out['imd_band_clean'].cat.codes.astype(float)
    codes[np.asarray(s == 'Unknown')] = np.nan
    out['imd_band_ord'] = codes
    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'n_submitted' in out:
        out['no_submission_yet'] = (out['n_submitted'].fillna(0) == 0).astype(int)
    if 'date_registration' in out:
        out['reg_date_missing'] = out['date_registration'].isna().astype(int)
    return out


def apply_landmark_risk_set(features: pd.DataFrame, cutoff_week: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff_day = 7 * cutoff_week
    x = features.drop(columns=[c for c in ['date_registration', 'date_unregistration'] if c in features],
                      errors='ignore').merge(registration, on=KEYS, how='left', validate='one_to_one')

    # Missing registration day is retained because OULAD contains legitimate missing values;
    # it is represented by the existing missingness flag rather than being treated as late registration.
    registered = x['date_registration'].isna() | (x['date_registration'] <= cutoff_day)
    outcome_not_yet_observed = x['date_unregistration'].isna() | (x['date_unregistration'] > cutoff_day)
    keep = registered & outcome_not_yet_observed

    audit = {
        'cutoff_week': cutoff_week,
        'cutoff_day': cutoff_day,
        'n_before': int(len(x)),
        'n_excluded_not_registered': int((~registered).sum()),
        'n_excluded_already_withdrawn': int((registered & ~outcome_not_yet_observed).sum()),
        'n_landmark_risk_set': int(keep.sum()),
        'positive_rate_landmark': float(x.loc[keep, 'at_risk'].mean()),
    }
    out = x.loc[keep].copy()
    out['landmark_day'] = cutoff_day
    assert (out['date_unregistration'].isna() | (out['date_unregistration'] > cutoff_day)).all()
    return out, audit


def fit_calibrator(raw_calibration_probability: Sequence[float], y_calibration: Sequence[int]):
    p = np.asarray(raw_calibration_probability, dtype=float)
    y = np.asarray(y_calibration, dtype=int)
    if len(p) >= 30 and len(np.unique(y)) == 2:
        model = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        return ('isotonic', model)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(p.reshape(-1, 1), y)
    return ('platt', model)


def apply_calibrator(calibrator, raw_probability: Sequence[float]) -> np.ndarray:
    kind, model = calibrator
    p = np.asarray(raw_probability, dtype=float)
    if kind == 'isotonic':
        return np.asarray(model.predict(p), dtype=float)
    return np.asarray(model.predict_proba(p.reshape(-1, 1))[:, 1], dtype=float)


@dataclass
class CalibratedBinaryPredictor:
    estimator: Any
    calibrator: Any
    threshold: float = 0.50

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(X)[:, 1]
        p1 = np.clip(apply_calibrator(self.calibrator, raw), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def build_models() -> dict[str, Any]:
    return {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)),
        ]),
        'rf': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=SEED
        ),
        'hgb': HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=SEED
        ),
    }


EXCLUDE = set(KEYS) | {
    'at_risk', 'final_result', 'cutoff_week', 'landmark_day', 'date_unregistration',
    'imd_band_clean', 'imd_band_ord', 'split', 'fold',
} | set(PROTECTED) | set(CONTEXT)


IMPUTE_COLS = ['date_registration', 'mean_score', 'weighted_mean_score', 'late_rate']


def cluster_resample_indices(df: pd.DataFrame, cluster_col: str, rng: np.random.Generator) -> np.ndarray:
    clusters = df[cluster_col].dropna().unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    groups = df.groupby(cluster_col, sort=False).indices
    return np.concatenate([np.asarray(groups[g], dtype=int) for g in sampled])


def binary_rates(y: Sequence[int], pred: Sequence[int]) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    pos = y == 1
    neg = y == 0
    return {
        'flag_rate': float(pred.mean()) if len(pred) else np.nan,
        'tpr': float(pred[pos].mean()) if pos.any() else np.nan,
        'fpr': float(pred[neg].mean()) if neg.any() else np.nan,
    }


def subgroup_rate_gap(df: pd.DataFrame, group_mask_a: np.ndarray, group_mask_b: np.ndarray,
                      pred_col: str, metric: str = 'fpr') -> float:
    a = df.loc[group_mask_a]
    b = df.loc[group_mask_b]
    ra = binary_rates(a['at_risk'], a[pred_col])[metric]
    rb = binary_rates(b['at_risk'], b[pred_col])[metric]
    return float(ra - rb)


def clustered_gap_ci(df: pd.DataFrame, group_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
                     pred_col: str, metric: str = 'fpr', n_boot: int = 1000,
                     seed: int = SEED) -> dict[str, float]:
    d = df.reset_index(drop=True).copy()
    a, b = group_fn(d)
    observed = subgroup_rate_gap(d, a, b, pred_col, metric)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = cluster_resample_indices(d, 'id_student', rng)
        z = d.iloc[idx].reset_index(drop=True)
        za, zb = group_fn(z)
        val = subgroup_rate_gap(z, za, zb, pred_col, metric)
        if np.isfinite(val):
            draws.append(val)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {'gap': observed, 'lo': float(lo), 'hi': float(hi), 'n_boot_valid': len(draws)}


def imd_extremes(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return d['imd_band_ord'].isin([0, 1, 2]).to_numpy(), d['imd_band_ord'].isin([7, 8, 9]).to_numpy()


def threshold_report(df: pd.DataFrame, p_col: str, threshold: float) -> dict[str, float]:
    y = df['at_risk'].to_numpy(int)
    p = df[p_col].to_numpy(float)
    pred = (p >= threshold).astype(int)
    r = binary_rates(y, pred)
    return {
        'threshold': float(threshold),
        'n': len(df),
        'flagged': int(pred.sum()),
        'flag_rate': r['flag_rate'],
        'tpr': r['tpr'], 'fpr': r['fpr'],
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
    }


def select_recall_constrained_threshold(calib: pd.DataFrame, p_col: str,
                                        min_recall: float = 0.80) -> float:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 199), calib[p_col].to_numpy()])
    feasible = []
    for t in candidates:
        r = threshold_report(calib, p_col, float(t))
        if r['tpr'] >= min_recall:
            feasible.append(r)
    if not feasible:
        raise RuntimeError('No threshold satisfies the requested recall on calibration data.')
    return float(max(feasible, key=lambda x: (x['precision'], x['threshold']))['threshold'])


def select_budget_threshold(calib: pd.DataFrame, p_col: str, budget_fraction: float) -> float:
    if not 0 < budget_fraction < 1:
        raise ValueError('budget_fraction must be between zero and one')
    return float(np.quantile(calib[p_col], 1.0 - budget_fraction))


def choose_threshold_for_target_fpr(calib_group: pd.DataFrame, p_col: str, target_fpr: float) -> float:
    negatives = calib_group.loc[calib_group['at_risk'].eq(0), p_col].to_numpy(float)
    if len(negatives) < 20:
        return np.nan
    # P(score >= t | y=0) = target_fpr
    return float(np.quantile(negatives, 1.0 - target_fpr))


def evaluate_policy(w: int, model: str = 'hgb', min_recall: float = 0.80,
                    budget_fraction: float = 0.25) -> pd.DataFrame:
    d = load_any(PROC / f'predictions_week{w}_all_splits')
    p_col = f'p_{model}'
    calib = d[d['split'].eq('calib')].copy()
    test = d[d['split'].eq('test')].copy()

    t_recall = select_recall_constrained_threshold(calib, p_col, min_recall)
    t_budget = select_budget_threshold(calib, p_col, budget_fraction)
    rows = []
    for policy, t in [('fixed_0.5', 0.5), ('recall_constrained', t_recall), ('budget', t_budget)]:
        rows.append({'cutoff_week': w, 'policy': policy, **threshold_report(test, p_col, t)})

    # Diagnostic only: thresholds are learned on calibration negatives, then frozen for test.
    c = clean_imd_band(calib)
    te = clean_imd_band(test)
    c_dep = c[c['imd_band_ord'].isin([0, 1, 2])]
    c_aff = c[c['imd_band_ord'].isin([7, 8, 9])]
    base_target = threshold_report(calib, p_col, t_recall)['fpr']
    t_dep = choose_threshold_for_target_fpr(c_dep, p_col, base_target)
    t_aff = choose_threshold_for_target_fpr(c_aff, p_col, base_target)
    for label, subset, t in [
        ('equal_fpr_diagnostic_deprived', te[te['imd_band_ord'].isin([0, 1, 2])], t_dep),
        ('equal_fpr_diagnostic_affluent', te[te['imd_band_ord'].isin([7, 8, 9])], t_aff),
    ]:
        if np.isfinite(t):
            rows.append({'cutoff_week': w, 'policy': label, **threshold_report(subset, p_col, t)})
    return pd.DataFrame(rows)


def feature_groups(features: Sequence[str]) -> dict[str, list[str]]:
    groups = {
        'engagement': [c for c in features if c.startswith('clicks_') or c in {
            'active_days', 'active_weeks', 'distinct_sites', 'clicks_per_active_day'}],
        'assessment': [c for c in features if c in {
            'n_submitted', 'n_due', 'submission_gap', 'mean_score',
            'weighted_mean_score', 'late_rate', 'no_submission_yet'}],
        'registration_load': [c for c in features if c in {
            'date_registration', 'reg_date_missing', 'num_of_prev_attempts', 'studied_credits'}],
    }
    used = set().union(*groups.values())
    for c in features:
        if c not in used:
            groups[f'singleton::{c}'] = [c]
    return {k: v for k, v in groups.items() if v}


def calibrated_predictor_from_bundle(bundle: Mapping[str, Any], model: str) -> CalibratedBinaryPredictor:
    item = bundle['models'][model]
    return CalibratedBinaryPredictor(item['estimator'], item['calibrator'], bundle.get('threshold', 0.5))


def model_agnostic_calibrated_shap(bundle: Mapping[str, Any], model: str,
                                   train_x: pd.DataFrame, eval_x: pd.DataFrame,
                                   background_n: int = 200):
    import shap
    features = list(bundle['features'])
    predictor = calibrated_predictor_from_bundle(bundle, model)
    background = train_x[features].sample(min(background_n, len(train_x)), random_state=SEED)
    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(
        lambda a: predictor.predict_proba(pd.DataFrame(a, columns=features))[:, 1],
        masker,
        algorithm='permutation',
        feature_names=features,
    )
    return explainer(eval_x[features])


def grouped_deletion_aopc(bundle: Mapping[str, Any], model: str,
                          train_x: pd.DataFrame, eval_x: pd.DataFrame,
                          shap_values: np.ndarray, n_donors: int = 5,
                          seed: int = SEED) -> np.ndarray:
    features = list(bundle['features'])
    groups = feature_groups(features)
    group_names = list(groups)
    positions = {c: i for i, c in enumerate(features)}
    group_importance = np.column_stack([
        np.abs(shap_values[:, [positions[c] for c in groups[g]]]).sum(axis=1)
        for g in group_names
    ])
    order = np.argsort(-group_importance, axis=1)

    predictor = calibrated_predictor_from_bundle(bundle, model)
    factual = eval_x[features].reset_index(drop=True)
    train = train_x[features].reset_index(drop=True)
    p0 = predictor.predict_proba(factual)[:, 1]
    predicted_class = (p0 >= predictor.threshold).astype(int)
    conf0 = np.where(predicted_class == 1, p0, 1.0 - p0)

    rng = np.random.default_rng(seed)
    row_aopc = np.zeros(len(factual), dtype=float)
    for _ in range(n_donors):
        donor_idx = rng.integers(0, len(train), size=len(factual))
        donor = train.iloc[donor_idx].reset_index(drop=True)
        perturbed = factual.copy()
        drops = []
        for step in range(len(group_names)):
            for i in range(len(perturbed)):
                cols = groups[group_names[order[i, step]]]
                perturbed.loc[i, cols] = donor.loc[i, cols].to_numpy()
            p = predictor.predict_proba(perturbed)[:, 1]
            conf = np.where(predicted_class == 1, p, 1.0 - p)
            drops.append(conf0 - conf)
        row_aopc += np.mean(np.column_stack(drops), axis=1)
    return row_aopc / n_donors


def common_support(df: pd.DataFrame, group_col: str, risk_col: str) -> pd.DataFrame:
    levels = list(pd.Series(df[group_col].dropna().unique()))
    if len(levels) != 2:
        raise ValueError(f'{group_col} must have exactly two levels')
    lo = max(df.loc[df[group_col].eq(g), risk_col].min() for g in levels)
    hi = min(df.loc[df[group_col].eq(g), risk_col].max() for g in levels)
    return df[df[risk_col].between(lo, hi, inclusive='both')].copy()


def risk_band_standardised_gap(df: pd.DataFrame, outcome_col: str, group_col: str,
                               risk_col: str, n_bins: int) -> dict[str, float]:
    d = common_support(df.dropna(subset=[outcome_col, group_col, risk_col]), group_col, risk_col)
    levels = sorted(d[group_col].unique())
    d['risk_band'] = pd.qcut(d[risk_col], q=n_bins, duplicates='drop')
    pieces = []
    for _, z in d.groupby('risk_band', observed=True):
        if z[group_col].nunique() < 2:
            continue
        counts = z.groupby(group_col).size()
        if counts.min() < 10:
            continue
        means = z.groupby(group_col)[outcome_col].mean()
        pieces.append((len(z), float(means.loc[levels[1]] - means.loc[levels[0]])))
    if not pieces:
        return {'gap': np.nan, 'n_common_support': len(d), 'bands_used': 0}
    weight = np.asarray([n for n, _ in pieces], dtype=float)
    gaps = np.asarray([g for _, g in pieces], dtype=float)
    return {'gap': float(np.average(gaps, weights=weight)),
            'n_common_support': len(d), 'bands_used': len(pieces)}


def matched_risk_gap(df: pd.DataFrame, outcome_col: str, group_col: str,
                     risk_col: str, caliper: float = 0.02) -> dict[str, float]:
    d = common_support(df.dropna(subset=[outcome_col, group_col, risk_col]), group_col, risk_col)
    levels = sorted(d[group_col].unique())
    a = d[d[group_col].eq(levels[0])].reset_index(drop=True)
    b = d[d[group_col].eq(levels[1])].reset_index(drop=True)
    if min(len(a), len(b)) == 0:
        return {'gap': np.nan, 'matched_pairs': 0}
    nn = NearestNeighbors(n_neighbors=1).fit(b[[risk_col]])
    distance, index = nn.kneighbors(a[[risk_col]])
    keep = distance[:, 0] <= caliper
    if keep.sum() == 0:
        return {'gap': np.nan, 'matched_pairs': 0}
    # level 1 minus level 0, consistent with the standardisation function above.
    gap = (b.iloc[index[keep, 0]][outcome_col].to_numpy() - a.loc[keep, outcome_col].to_numpy()).mean()
    return {'gap': float(gap), 'matched_pairs': int(keep.sum())}


def risk_adjustment_sensitivity(df: pd.DataFrame, outcome_col: str, group_col: str,
                                risk_col: str) -> pd.DataFrame:
    rows = []
    raw = df.groupby(group_col)[outcome_col].mean().sort_index()
    rows.append({'method': 'raw', 'setting': 'none', 'gap': float(raw.iloc[1] - raw.iloc[0])})
    for b in [4, 5, 10]:
        r = risk_band_standardised_gap(df, outcome_col, group_col, risk_col, b)
        rows.append({'method': 'risk-band standardisation', 'setting': f'{b} bins', **r})
    for caliper in [0.01, 0.02, 0.05]:
        r = matched_risk_gap(df, outcome_col, group_col, risk_col, caliper)
        rows.append({'method': 'nearest-risk matching', 'setting': f'caliper={caliper}', **r})
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def validate_counterfactual_state(factual: pd.Series, counterfactual: pd.Series,
                                  cutoff_week: int, atol: float = 1e-6) -> list[str]:
    violations = []
    cumulative = [c for c in factual.index if c.startswith('clicks_')]
    cumulative += [c for c in ['active_days', 'active_weeks', 'distinct_sites', 'n_submitted']
                   if c in factual.index]
    for c in cumulative:
        if c in counterfactual and counterfactual[c] + atol < factual[c]:
            violations.append(f'{c} decreased despite being cumulative')

    if {'clicks_total', 'active_days', 'clicks_per_active_day'} <= set(counterfactual.index):
        expected = (counterfactual['clicks_total'] / counterfactual['active_days']
                    if counterfactual['active_days'] > 0 else 0.0)
        if not np.isclose(counterfactual['clicks_per_active_day'], expected, atol=atol):
            violations.append('clicks_per_active_day is inconsistent with clicks_total/active_days')

    type_cols = [c for c in counterfactual.index if c.startswith('clicks_') and c not in {
        'clicks_total', 'clicks_last_2w', 'clicks_first_half', 'clicks_second_half',
        'clicks_per_active_day'}]
    if 'clicks_total' in counterfactual and type_cols:
        if not np.isclose(counterfactual[type_cols].sum(), counterfactual['clicks_total'], atol=atol):
            violations.append('activity-type clicks do not sum to clicks_total')

    if {'clicks_first_half', 'clicks_second_half', 'clicks_total'} <= set(counterfactual.index):
        if not np.isclose(counterfactual['clicks_first_half'] + counterfactual['clicks_second_half'],
                          counterfactual['clicks_total'], atol=atol):
            violations.append('first-half + second-half clicks do not equal clicks_total')

    if {'n_due', 'n_submitted', 'submission_gap'} <= set(counterfactual.index):
        expected_gap = max(0.0, counterfactual['n_due'] - counterfactual['n_submitted'])
        if counterfactual['n_submitted'] > counterfactual['n_due'] + atol:
            violations.append('n_submitted exceeds n_due')
        if not np.isclose(counterfactual['submission_gap'], expected_gap, atol=atol):
            violations.append('submission_gap is inconsistent with n_due - n_submitted')

    if 'active_days' in counterfactual and counterfactual['active_days'] > 7 * cutoff_week + atol:
        violations.append('active_days exceeds the elapsed landmark window')
    if 'active_weeks' in counterfactual and counterfactual['active_weeks'] > cutoff_week + 1 + atol:
        violations.append('active_weeks exceeds the elapsed landmark window')
    return violations


def build_observed_delta_library(current: pd.DataFrame, future: pd.DataFrame,
                                 primitive_cumulative_features: Sequence[str]) -> pd.DataFrame:
    joined = current[KEYS + list(primitive_cumulative_features)].merge(
        future[KEYS + list(primitive_cumulative_features)], on=KEYS,
        suffixes=('_current', '_future'), validate='one_to_one')
    for c in primitive_cumulative_features:
        joined[f'delta_{c}'] = joined[f'{c}_future'] - joined[f'{c}_current']
    delta_cols = [f'delta_{c}' for c in primitive_cumulative_features]
    return joined[(joined[delta_cols] >= 0).all(axis=1)].copy()
