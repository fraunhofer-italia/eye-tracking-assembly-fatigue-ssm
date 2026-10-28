"""Leakage-controlled participant-wise SSM experiments for the fatigue manuscript.

This is an independent *offline evaluation* implementation of the manuscript's
150-second ocular window / 300-second SSS-aligned recursive state model. It is
NOT interchangeable with the 150s/30s real-time deployment model bundle.

Time series in the input CSV are already processed, causal ocular summaries.
Only train participants contribute to learned parameters or target interpolation.
Test SSS is used solely for scoring and optional *post-prediction* feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from numba import njit
except ImportError:  # Verified fallback (slower, but mathematically equivalent)
    njit = None

if njit is not None:
    @njit(cache=True)
    def _forest_batch_predict(X, left, right, features, thresholds, values):
        out = np.empty(X.shape[0], dtype=np.float64)
        for row in range(X.shape[0]):
            total = 0.0
            for tree in range(left.shape[0]):
                node = 0
                while left[tree, node] != -1:
                    column = features[tree, node]
                    node = left[tree, node] if X[row, column] <= thresholds[tree, node] else right[tree, node]
                total += values[tree, node]
            out[row] = total / left.shape[0]
        return out


def pack_extra_trees(model: Pipeline | None):
    """Optional exact fast inference path for 10s/30s grid evaluations."""
    if model is None or njit is None:
        return None
    forest = model.named_steps.get("regressor")
    if not isinstance(forest, ExtraTreesRegressor):
        return None
    trees = forest.estimators_
    max_nodes = max(tree.tree_.node_count for tree in trees)
    shape = (len(trees), max_nodes)
    left = np.full(shape, -1, dtype=np.int32)
    right = np.full(shape, -1, dtype=np.int32)
    features = np.full(shape, -2, dtype=np.int32)
    thresholds = np.zeros(shape, dtype=np.float64)
    values = np.zeros(shape, dtype=np.float64)
    for tree_id, estimator in enumerate(trees):
        tree = estimator.tree_
        count = tree.node_count
        left[tree_id, :count] = tree.children_left
        right[tree_id, :count] = tree.children_right
        features[tree_id, :count] = tree.feature
        thresholds[tree_id, :count] = tree.threshold
        values[tree_id, :count] = tree.value[:, 0, 0]
    return left, right, features, thresholds, values


def predict_transition(core, predictors: np.ndarray) -> np.ndarray:
    if core.packed_transition is None:
        return core.transition_model.predict(predictors)
    imputed = core.transition_model.named_steps["imputer"].transform(predictors)
    return _forest_batch_predict(np.asarray(imputed, dtype=np.float64), *core.packed_transition)

WINDOWS = (10, 30, 60, 150, 300)
CADENCES = (10, 30, 60, 150, 300)
REFERENCE_WINDOW = 150
REFERENCE_CADENCE = 300
FATIGUE_THRESHOLD = 4.0
SEED = 42
VARIANCE_FLOOR = 1e-4
INITIAL_VARIANCE = 1e-4
LOG_EPS = 1e-3
ONLINE_L2 = 0.1  # coefficient on the squared logit offset (not half the coefficient)
FEEDBACK_ORDINALS = (0, 4, 8)  # first, fifth, ninth observed assembly SSS
FOLDS = {
    1: ("2", "3", "9", "11"),
    2: ("8", "14", "16", "19"),
    3: ("7", "12", "15", "18"),
    4: ("1", "4", "13", "17", "21"),
    5: ("5", "6", "10", "20", "22"),
}
GEOMETRY = (
    "geo_eyelid aperture left [mm]_mean",
    "geo_eyelid aperture left [mm]_std",
    "geo_eyelid aperture right [mm]_mean",
    "geo_eyelid aperture right [mm]_std",
    "geo_vergence_deg_mean",
    "geo_vergence_deg_std",
    "eye_pose_std",
)
SCALE_FEATURES = frozenset((GEOMETRY[1], GEOMETRY[3]))
NORMALIZED = tuple(g + ("_logratio" if g in SCALE_FEATURES else "_delta") for g in GEOMETRY)
META_COLUMNS = (
    "participant_id", "session", "task", "prediction_time", "assembly_start_time",
    "SSS_score", "real_SSS_score", "is_eval_sss_window", "is_initial_state_anchor",
    "exclude_from_training_eval",
)
KEYS = ["participant_id", "session"]


def as_bool(value: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(value):
        return value.fillna(False).astype(bool)
    return value.astype(str).str.strip().str.lower().isin(("1", "true", "t", "yes", "y"))


def load_dataset(path: str | Path, windows: Iterable[int] = WINDOWS) -> pd.DataFrame:
    """Read only 10 metadata columns plus multiscale model inputs from 742-column CSV."""
    windows = sorted(set(int(w) for w in windows))
    if not windows or set(windows) - set(WINDOWS):
        raise ValueError(f"Unsupported feature windows: {windows}")
    needed = list(META_COLUMNS)
    for w in windows:
        needed += [f"feature_window_valid_w{w}s", f"perclos_mean_w{w}s_within"]
        needed += [f"{g}_w{w}s" for g in GEOMETRY]
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(needed) - set(header))
    if missing:
        raise ValueError(f"Required dataset columns are missing: {missing}")
    data = pd.read_csv(path, usecols=needed, low_memory=False)
    data["participant_id"] = data["participant_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    for key in ("session", "task"):
        data[key] = data[key].astype(str).str.lower().str.strip()
    for key in ("is_eval_sss_window", "is_initial_state_anchor", "exclude_from_training_eval"):
        data[key] = as_bool(data[key])
    for w in windows:
        data[f"feature_window_valid_w{w}s"] = as_bool(data[f"feature_window_valid_w{w}s"])
    for key in ["prediction_time", "assembly_start_time", "SSS_score", "real_SSS_score"]:
        data[key] = pd.to_numeric(data[key], errors="coerce")
    data = data.sort_values(KEYS + ["prediction_time"], kind="stable").reset_index(drop=True)
    return data


def audit_data(data: pd.DataFrame, windows: Iterable[int] = WINDOWS) -> dict:
    anchors = data[data.is_initial_state_anchor]
    evals = data[data.is_eval_sss_window]
    groups = data.groupby(KEYS).size()
    aud = {
        "input_rows": int(len(data)),
        "participants": int(data.participant_id.nunique()),
        "participant_sessions": int(len(groups)),
        "anchors": int(len(anchors)),
        "exact_sss_rows": int(len(evals)),
        "exact_sss_with_real_label": int(evals.real_SSS_score.notna().sum()),
        "label_prevalence_sss_ge_4": float((evals.real_SSS_score >= 4).mean()),
        "observed_sss_histogram": {str(int(k)): int(v) for k, v in evals.real_SSS_score.value_counts().sort_index().items()},
        "anchors_per_session_valid": bool((anchors.groupby(KEYS).size() == 1).all() and len(anchors.groupby(KEYS)) == len(groups)),
    }
    for w in windows:
        key = f"feature_window_valid_w{w}s"
        aud[f"valid_eval_rows_w{w}"] = int((evals[key] & evals.real_SSS_score.notna()).sum())
    return aud


def prepare_window(data: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Select valid causal window and session-anchor FSAN geometry; no test labels used."""
    valid = data[f"feature_window_valid_w{window_s}s"]
    keep = (data.is_initial_state_anchor | (data.task.eq("assembly_realtime") & ~data.exclude_from_training_eval))
    frame = data.loc[valid & keep].copy()
    frame["perclos"] = pd.to_numeric(frame[f"perclos_mean_w{window_s}s_within"], errors="coerce")
    for name in GEOMETRY:
        frame[name] = pd.to_numeric(frame[f"{name}_w{window_s}s"], errors="coerce")
    duplicated = frame.duplicated(KEYS + ["prediction_time"])
    if duplicated.any():
        raise ValueError("Duplicate participant/session/prediction_time rows")
    counts = frame[frame.is_initial_state_anchor].groupby(KEYS).size()
    if len(counts) != frame.groupby(KEYS).ngroups or not counts.eq(1).all():
        raise ValueError("Each participant-session must have exactly one valid initial SSS anchor")
    # Anchor is strictly pre-assembly. Normalization cannot access later test SSS.
    base = frame[frame.is_initial_state_anchor].set_index(KEYS)
    current_index = pd.MultiIndex.from_frame(frame[KEYS])
    for name in GEOMETRY:
        baseline = base[name].reindex(current_index).to_numpy(dtype=float)
        current = frame[name].to_numpy(dtype=float)
        if name in SCALE_FEATURES:
            # The manuscript specifies positive scale variables and a fixed small epsilon.
            valid_pair = (current >= 0) & (baseline >= 0)
            value = np.where(valid_pair, np.log((np.maximum(current, 0) + LOG_EPS) /
                                                (np.maximum(baseline, 0) + LOG_EPS)), np.nan)
            frame[name + "_logratio"] = value
        else:
            frame[name + "_delta"] = current - baseline
    frame["elapsed_minutes"] = np.maximum(0.0, (frame.prediction_time - frame.assembly_start_time) / 60.0)
    frame["afternoon"] = frame.session.eq("afternoon").astype(float)
    return frame.sort_values(KEYS + ["prediction_time"], kind="stable").reset_index(drop=True)


def build_sss_chain(frame: pd.DataFrame, cadence_s: int) -> pd.DataFrame:
    """Exact SSS endpoints + rounded-step support nodes from causal candidate rows.

    An interval of duration dt gets n=max(1, floor(dt/cadence + 0.5)) steps.
    This avoids spurious extra nodes for 300.1-second questionnaire intervals.
    The n-1 support requests are aligned backward from the next exact SSS.
    No future SSS values are accessed here (only timestamps and flags).
    """
    if cadence_s <= 0:
        raise ValueError("cadence_s must be positive")
    parts = []
    for name, group in frame.groupby(KEYS, sort=False):
        g = group.sort_values("prediction_time").copy()
        anchor = g[g.is_initial_state_anchor]
        endpoints = g[g.is_eval_sss_window & g.real_SSS_score.notna()]
        if len(anchor) != 1 or endpoints.empty:
            raise ValueError(f"Invalid anchor / exact questionnaire rows for {name}")
        start = float(anchor.iloc[0].prediction_time)
        keep = {int(anchor.index[0])}
        support_to_target: dict[int, float] = {}
        for idx, row in endpoints.iterrows():
            end = float(row.prediction_time)
            if end <= start:
                raise ValueError(f"Non-increasing questionnaire timestamps for {name}")
            steps = max(1, int(np.floor((end - start) / cadence_s + 0.5)))
            candidates = g[(g.prediction_time > start) & (g.prediction_time < end) &
                           ~g.is_eval_sss_window & ~g.is_initial_state_anchor]
            for k in range(1, steps):
                target = end - k * cadence_s
                if start < target < end and not candidates.empty:
                    pos = (candidates.prediction_time - target).abs().idxmin()
                    if pos not in keep:
                        keep.add(int(pos))
                        support_to_target[int(pos)] = float(target)
            keep.add(int(idx))
            start = end
        selected = g.loc[sorted(keep, key=lambda idx: g.at[idx, "prediction_time"])].copy()
        selected["is_support"] = ~selected.is_initial_state_anchor & ~selected.is_eval_sss_window
        selected["support_target_time"] = [support_to_target.get(idx, np.nan) for idx in selected.index]
        parts.append(selected)
    chain = pd.concat(parts).sort_values(KEYS + ["prediction_time"], kind="stable").reset_index(drop=True)
    chain["depth"] = chain.groupby(KEYS, sort=False).cumcount()
    chain["eval_ordinal"] = np.nan
    for _, idx in chain.groupby(KEYS, sort=False).groups.items():
        idx = list(idx)
        observed = [i for i in idx if bool(chain.at[i, "is_eval_sss_window"])]
        for ordinal, i in enumerate(observed):
            chain.at[i, "eval_ordinal"] = ordinal
    return chain


def training_targets(chain: pd.DataFrame, *, use_interpolation: bool = True) -> np.ndarray:
    """Create train-only supervision at retained support points.

    Never invoke this on held-out participants: it interpolates neighboring
    *observed* SSS labels and those are not allowed in blind test inference.
    """
    y = np.full(len(chain), np.nan, dtype=float)
    for _, group in chain.groupby(KEYS, sort=False):
        known = group[group.is_initial_state_anchor | group.is_eval_sss_window]
        known = known[known.real_SSS_score.notna()]
        if known.empty:
            continue
        measured = known.real_SSS_score.to_numpy(dtype=float)
        times = known.prediction_time.to_numpy(dtype=float)
        idx = group.index.to_numpy(dtype=int)
        if use_interpolation:
            y[idx] = np.interp(group.prediction_time.to_numpy(dtype=float), times, measured)
        else:
            y[known.index.to_numpy(dtype=int)] = measured
    return y


def parent_positions(chain: pd.DataFrame) -> np.ndarray:
    parent = np.full(len(chain), -1, dtype=int)
    for _, group in chain.groupby(KEYS, sort=False):
        idx = group.index.to_numpy(dtype=int)
        parent[idx[1:]] = idx[:-1]
    return parent


@dataclass(frozen=True)
class Variant:
    name: str = "reference"
    window_s: int = REFERENCE_WINDOW
    cadence_s: int = REFERENCE_CADENCE
    obs: str = "fsan"  # fsan | raw | none
    transition: str = "extra_trees"  # extra_trees | linear | persistence
    use_perclos: bool = True
    fusion: str = "both"  # both | observation_only | transition_only
    known_initial: bool = True
    cal: str = "session_time"  # session_time | session_only | time_only | latent_only
    interpolated_training: bool = True
    fatigue_threshold: float = FATIGUE_THRESHOLD
    feedback: str = "none"  # none | state | intercept | both
    feedback_ordinals: tuple[int, ...] = FEEDBACK_ORDINALS
    model_type: str = "state_space"  # state_space | context_only | static_geometry


@dataclass
class Core:
    variant: Variant
    obs_model: Pipeline | None
    transition_model: Pipeline | None
    mean: float
    std: float
    R: float
    Q: float
    n_transition_pairs: int
    n_obs_train: int
    importance_perclos: float
    packed_transition: tuple | None = None


def feature_names(variant: Variant) -> list[str]:
    if variant.obs == "none":
        return []
    if variant.obs == "raw":
        return list(GEOMETRY)
    return list(NORMALIZED)


def observation_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])


def transition_pipeline(kind: str, n_trees: int = 200) -> Pipeline | None:
    if kind == "persistence":
        return None
    predictor = (ExtraTreesRegressor(n_estimators=n_trees, max_depth=4,
                                     min_samples_leaf=5, n_jobs=1,
                                     random_state=SEED) if kind == "extra_trees"
                 else Ridge(alpha=1.0))
    transforms = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "linear":
        transforms.append(("scaler", StandardScaler()))
    transforms.append(("regressor", predictor))
    return Pipeline(transforms)


def fit_core(train: pd.DataFrame, variant: Variant, n_trees: int = 200) -> Core:
    y = training_targets(train, use_interpolation=variant.interpolated_training)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if not np.isfinite(std) or std < 1e-9:
        std = 1.0
    standardized = (y - mean) / std
    assembly = ~train.is_initial_state_anchor.to_numpy(dtype=bool)
    features = feature_names(variant)
    obs_model = None
    R = 1.0
    n_obs = 0
    if variant.fusion != "transition_only" and features:
        mask = assembly & np.isfinite(standardized)
        obs_model = observation_pipeline()
        obs_model.fit(train.loc[mask, features], standardized[mask])
        residuals = standardized[mask] - obs_model.predict(train.loc[mask, features])
        R = float(np.var(residuals) + VARIANCE_FLOOR)
        n_obs = int(mask.sum())
    transition_model = None
    Q = 1.0
    pairs = 0
    importance = np.nan
    if variant.fusion != "observation_only":
        parent = parent_positions(train)
        use = assembly & (parent >= 0) & np.isfinite(standardized)
        use &= np.isfinite(standardized[np.maximum(parent, 0)])
        targets = standardized[use]
        previous = standardized[parent[use]]
        arrays = [previous]
        if variant.use_perclos:
            arrays.append(train.perclos.to_numpy(dtype=float)[use])
        X = np.column_stack(arrays)
        pairs = len(targets)
        if pairs < 5:
            raise ValueError("Too few valid training transition pairs")
        if variant.transition == "persistence":
            fitted = previous
        else:
            transition_model = transition_pipeline(variant.transition, n_trees)
            transition_model.fit(X, targets)
            fitted = transition_model.predict(X)
            if variant.transition == "extra_trees" and variant.use_perclos:
                importance = float(transition_model.named_steps["regressor"].feature_importances_[1])
        Q = float(np.var(targets - fitted) + VARIANCE_FLOOR)
    return Core(variant, obs_model, transition_model, mean, std, R, Q, pairs, n_obs, importance,
                pack_extra_trees(transition_model))


def calibration_inputs(frame: pd.DataFrame, latent: np.ndarray | None, mode: str) -> np.ndarray:
    a = frame.afternoon.to_numpy(dtype=float)
    t = frame.elapsed_minutes.to_numpy(dtype=float)
    if mode == "context_only":
        return np.column_stack((a, t, a * t))
    if mode == "static_geometry":
        raise ValueError("Use direct normalized geometry columns for the static model")
    z = np.asarray(latent, dtype=float)
    if mode == "latent_only":
        return z[:, None]
    if mode == "session_only":
        return np.column_stack((z, a, z * a))
    if mode == "time_only":
        return np.column_stack((z, t, z * t))
    if mode != "session_time":
        raise ValueError(mode)
    return np.column_stack((z, a, z * a, t, z * t))


def calibrator_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", solver="liblinear",
                                          max_iter=1000, random_state=SEED)),
    ])


def infer_sequence(frame: pd.DataFrame, core: Core, calibrator: Pipeline | None = None,
                   *, feedback: str = "none", feedback_ordinals: tuple[int, ...] = FEEDBACK_ORDINALS,
                   threshold: float = FATIGUE_THRESHOLD) -> pd.DataFrame:
    """Filter chronological chains; the current SSS is NEVER read before its prediction.

    All sessions are evaluated by sequence depth for vectorized tree predictions.
    Observed SSS is consulted only for initial anchors and feedback AFTER scoring.
    """
    work = frame.reset_index(drop=True).copy()
    n = len(work)
    par = parent_positions(work)
    obs = np.full(n, np.nan)
    if core.obs_model is not None:
        obs = core.obs_model.predict(work[feature_names(core.variant)]).astype(float)
    state = np.full(n, np.nan)
    prior = np.full(n, np.nan)
    gain = np.full(n, np.nan)
    variance = np.full(n, np.nan)
    state_for_next = np.full(n, np.nan)
    variance_for_next = np.full(n, np.nan)
    probability = np.full(n, np.nan)
    base_probability = np.full(n, np.nan)
    reset_after = np.zeros(n, dtype=bool)
    offset_before = np.zeros(n)
    key_ids = list(zip(work.participant_id.astype(str), work.session.astype(str)))
    offsets = {key: 0.0 for key in set(key_ids)}
    feedback_history: dict[tuple[str, str], list[tuple[float, int]]] = {key: [] for key in offsets}
    perclos = work.perclos.to_numpy(dtype=float)
    observed = work.real_SSS_score.to_numpy(dtype=float)
    anchor = work.is_initial_state_anchor.to_numpy(dtype=bool)
    ordinals = work.eval_ordinal.to_numpy(dtype=float)
    elapsed_levels = work.depth.to_numpy(dtype=int)
    for level in range(int(elapsed_levels.max()) + 1):
        positions = np.flatnonzero(elapsed_levels == level)
        if not len(positions):
            continue
        if level == 0:
            # Only the pre-assembly SSS is legally visible during blind initialization.
            for p in positions:
                if core.variant.known_initial and anchor[p] and np.isfinite(observed[p]):
                    state[p] = (observed[p] - core.mean) / core.std
                    variance[p] = INITIAL_VARIANCE
                else:
                    state[p] = 0.0  # neutral training-fold mean (no test questionnaire)
                    variance[p] = 1.0
        elif core.variant.fusion == "observation_only":
            state[positions] = obs[positions]
            variance[positions] = core.R
        else:
            parents = par[positions]
            previous = state_for_next[parents]
            if core.variant.transition == "persistence":
                prior[positions] = previous
            else:
                predictors = [previous]
                if core.variant.use_perclos:
                    predictors.append(perclos[positions])
                prior[positions] = predict_transition(core, np.column_stack(predictors))
            predvar = variance_for_next[parents] + core.Q
            if core.variant.fusion == "transition_only":
                state[positions] = prior[positions]
                variance[positions] = predvar
                gain[positions] = 0.0
            else:
                gain[positions] = predvar / (predvar + core.R)
                state[positions] = prior[positions] + gain[positions] * (obs[positions] - prior[positions])
                variance[positions] = (1.0 - gain[positions]) * predvar
        state_for_next[positions] = state[positions]
        variance_for_next[positions] = variance[positions]
        if calibrator is not None:
            Xcal = calibration_inputs(work.iloc[positions], state[positions], core.variant.cal)
            base_probability[positions] = calibrator.predict_proba(Xcal)[:, 1]
            for p in positions:
                key = key_ids[p]
                offset = offsets[key] if feedback in ("intercept", "both") else 0.0
                offset_before[p] = offset
                probability[p] = expit(logit(np.clip(base_probability[p], 1e-9, 1-1e-9)) + offset)
                # Crucial order: predict and score before using observed feedback.
                is_feedback = bool(work.at[p, "is_eval_sss_window"] and
                                   np.isfinite(ordinals[p]) and int(ordinals[p]) in feedback_ordinals)
                if is_feedback and feedback != "none" and np.isfinite(observed[p]):
                    if feedback in ("state", "both"):
                        state_for_next[p] = (observed[p] - core.mean) / core.std
                        variance_for_next[p] = INITIAL_VARIANCE
                        reset_after[p] = True
                    if feedback in ("intercept", "both"):
                        feedback_history[key].append((float(logit(np.clip(base_probability[p], 1e-9, 1-1e-9))),
                                                      int(observed[p] >= threshold)))
                        offsets[key] = fit_online_offset(feedback_history[key], ONLINE_L2)
    work["latent"] = state
    work["latent_variance"] = variance
    work["transition_prior"] = prior
    work["ridge_observation"] = obs
    work["fusion_gain"] = gain
    work["base_probability"] = base_probability
    work["probability"] = probability
    work["online_logit_offset_before"] = offset_before
    work["reset_after_scoring"] = reset_after
    work["y_pred_sss"] = state * core.std + core.mean
    work["Q"] = core.Q
    work["R"] = core.R
    work["transition_perclos_importance"] = core.importance_perclos
    return work


def fit_online_offset(history: list[tuple[float, int]], lam: float = ONLINE_L2) -> float:
    """Minimize mean BCE(logit+offset) + lam*offset**2 with Newton updates."""
    base = np.asarray([h[0] for h in history])
    targets = np.asarray([h[1] for h in history])
    delta = 0.0
    for _ in range(70):
        probs = expit(base + delta)
        grad = float(np.mean(probs - targets) + 2 * lam * delta)
        hessian = float(np.mean(probs * (1 - probs)) + 2 * lam)
        new = float(np.clip(delta - grad / max(hessian, 1e-8), -6, 6))
        if abs(new - delta) < 1e-10:
            return new
        delta = new
    return delta


def train_predict(train: pd.DataFrame, test: pd.DataFrame, variant: Variant,
                  n_trees: int = 200) -> tuple[pd.DataFrame, Core | None]:
    """Fit exclusively on training people; predict held-out people."""
    if variant.model_type in ("context_only", "static_geometry"):
        model = calibrator_pipeline()
        observed = train.is_eval_sss_window & train.real_SSS_score.notna()
        if variant.model_type == "context_only":
            train_x = calibration_inputs(train.loc[observed], None, "context_only")
            test_x = calibration_inputs(test, None, "context_only")
        else:
            train_x = train.loc[observed, list(NORMALIZED)]
            test_x = test[list(NORMALIZED)]
        train_y = (train.loc[observed, "real_SSS_score"] >= variant.fatigue_threshold).astype(int)
        model.fit(train_x, train_y)
        output = test.copy()
        output["probability"] = model.predict_proba(test_x)[:, 1]
        output["latent"] = np.nan
        output["transition_prior"] = np.nan
        output["ridge_observation"] = np.nan
        output["fusion_gain"] = np.nan
        output["Q"] = np.nan
        output["R"] = np.nan
        output["y_pred_sss"] = np.nan
        output["reset_after_scoring"] = False
        output["online_logit_offset_before"] = 0.0
        output["transition_perclos_importance"] = np.nan
        output["training_target_mean"] = np.nan
        output["training_target_std"] = np.nan
        return output, None
    core = fit_core(train, variant, n_trees)
    train_latents = infer_sequence(train, core).latent.to_numpy()
    is_eval = train.is_eval_sss_window.to_numpy(dtype=bool) & train.real_SSS_score.notna().to_numpy()
    calibrator = calibrator_pipeline()
    cal_x = calibration_inputs(train.loc[is_eval], train_latents[is_eval], variant.cal)
    cal_y = (train.loc[is_eval, "real_SSS_score"] >= variant.fatigue_threshold).astype(int)
    calibrator.fit(cal_x, cal_y)
    output = infer_sequence(test, core, calibrator, feedback=variant.feedback,
                            feedback_ordinals=variant.feedback_ordinals, threshold=variant.fatigue_threshold)
    output["training_target_mean"] = core.mean
    output["training_target_std"] = core.std
    return output, core


def exact_predictions(output: pd.DataFrame, variant: Variant, fold: int) -> pd.DataFrame:
    exact = output[output.is_eval_sss_window & output.real_SSS_score.notna()].copy()
    exact["experiment"] = variant.name
    exact["fold"] = fold
    exact["window_s"] = variant.window_s
    exact["cadence_s"] = variant.cadence_s
    exact["threshold_sss"] = variant.fatigue_threshold
    exact["y_true"] = (exact.real_SSS_score >= variant.fatigue_threshold).astype(int)
    exact["y_pred"] = (exact.probability >= 0.5).astype(int)
    cols = ["experiment", "fold", "participant_id", "session", "prediction_time", "eval_ordinal",
            "window_s", "cadence_s", "threshold_sss", "real_SSS_score", "y_true", "probability", "y_pred",
            "y_pred_sss", "latent", "transition_prior", "ridge_observation", "fusion_gain", "Q", "R",
            "transition_perclos_importance", "reset_after_scoring", "online_logit_offset_before", "perclos",
            "training_target_mean", "training_target_std"]
    return exact[cols]


def scores(rows: pd.DataFrame) -> dict:
    y = rows.y_true.to_numpy(dtype=int)
    p = rows.probability.to_numpy(dtype=float)
    yp = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (yp == 0)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    tp = int(((y == 1) & (yp == 1)).sum())
    out = {
        "rows": int(len(rows)), "participants": int(rows.participant_id.nunique()),
        "positive_rate": float(y.mean()), "accuracy": float(accuracy_score(y, yp)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "F1": float(f1_score(y, yp, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
        "AP": float(average_precision_score(y, p)) if len(set(y)) == 2 else np.nan,
        "Brier": float(brier_score_loss(y, p)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
    valid = np.isfinite(rows.y_pred_sss.to_numpy(dtype=float))
    if valid.any():
        target = rows.real_SSS_score.to_numpy(dtype=float)[valid]
        pred = rows.y_pred_sss.to_numpy(dtype=float)[valid]
        out["SSS_MAE"] = float(np.mean(np.abs(target - pred)))
        out["SSS_RMSE"] = float(np.sqrt(np.mean((target - pred) ** 2)))
        out["SSS_R2"] = float(r2_score(target, pred)) if len(target) > 1 else np.nan
    else:
        out["SSS_MAE"] = out["SSS_RMSE"] = out["SSS_R2"] = np.nan
    if "training_target_std" in rows and rows.training_target_std.notna().any():
        norm = rows.training_target_std.to_numpy(dtype=float)
        true_z = (rows.real_SSS_score.to_numpy(dtype=float) - rows.training_target_mean.to_numpy(dtype=float)) / norm
        for label, column in (("state", "latent"), ("prior", "transition_prior"),
                              ("ridge", "ridge_observation")):
            pred_z = rows[column].to_numpy(dtype=float)
            supported = np.isfinite(pred_z) & np.isfinite(true_z)
            if supported.any():
                out[f"{label}_scaled_RMSE"] = float(np.sqrt(np.mean((pred_z[supported] - true_z[supported]) ** 2)))
                out[f"{label}_scaled_R2"] = float(r2_score(true_z[supported], pred_z[supported]))
    out["Q_mean"] = float(rows.Q.mean()) if rows.Q.notna().any() else np.nan
    out["R_mean"] = float(rows.R.mean()) if rows.R.notna().any() else np.nan
    out["K_median"] = float(rows.fusion_gain.median()) if rows.fusion_gain.notna().any() else np.nan
    out["perclos_importance_mean"] = (float(rows.transition_perclos_importance.mean())
                                      if rows.transition_perclos_importance.notna().any() else np.nan)
    return out


def summary_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for (experiment, fold), group in predictions.groupby(["experiment", "fold"], sort=False):
        fold_rows.append({"experiment": experiment, "fold": int(fold), **scores(group)})
    fold_table = pd.DataFrame(fold_rows)
    summary_rows = []
    for experiment, group in predictions.groupby("experiment", sort=False):
        sub = fold_table[fold_table.experiment == experiment]
        stats = scores(group)
        stats["experiment"] = experiment
        stats["fold_F1_mean"] = float(sub.F1.mean())
        stats["fold_F1_sd"] = float(sub.F1.std(ddof=1)) if len(sub) > 1 else np.nan
        stats["fold_AUC_mean"] = float(sub.ROC_AUC.mean())
        stats["fold_AUC_sd"] = float(sub.ROC_AUC.std(ddof=1)) if len(sub) > 1 else np.nan
        summary_rows.append(stats)
    return fold_table, pd.DataFrame(summary_rows)


def bootstrap_paired_f1(predictions: pd.DataFrame, reference: str = "reference",
                        n_boot: int = 10_000, seed: int = SEED) -> pd.DataFrame:
    """10k paired participant-cluster percentile bootstraps of pooled F1 differences."""
    base = predictions[predictions.experiment == reference].copy()
    if base.empty or n_boot <= 0:
        return pd.DataFrame()
    out = []
    for name, group in predictions.groupby("experiment", sort=False):
        if name == reference or group.threshold_sss.iloc[0] != base.threshold_sss.iloc[0]:
            continue
        keys = ["participant_id", "session", "prediction_time"]
        merged = base.merge(group, on=keys, validate="one_to_one", suffixes=("_ref", "_other"))
        if len(merged) != len(base) or len(merged) != len(group):
            continue  # different evaluation endpoints are not a matched comparison
        if not (merged.y_true_ref.to_numpy() == merged.y_true_other.to_numpy()).all():
            raise ValueError("Paired comparison has inconsistent labels")
        # Per-participant confusion counts, then sample participant IDs with replacement.
        unique = sorted(merged.participant_id.unique())
        counts = np.zeros((2, len(unique), 3), dtype=np.float64)
        for j, pid in enumerate(unique):
            person = merged[merged.participant_id == pid]
            y = person.y_true_ref.to_numpy(dtype=int)
            for k, col in enumerate(("y_pred_ref", "y_pred_other")):
                yp = person[col].to_numpy(dtype=int)
                counts[k, j] = [(y & yp).sum(), ((1-y) & yp).sum(), (y & (1-yp)).sum()]
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(unique), (n_boot, len(unique)))
        sampled = counts[:, samples].sum(axis=2)
        f1 = 2 * sampled[:, :, 0] / np.maximum(2 * sampled[:, :, 0] + sampled[:, :, 1] + sampled[:, :, 2], 1e-12)
        differences = f1[0] - f1[1]
        reference_f1 = f1_score(merged.y_true_ref, merged.y_pred_ref, zero_division=0)
        other_f1 = f1_score(merged.y_true_other, merged.y_pred_other, zero_division=0)
        out.append({"reference": reference, "comparison": name, "participants": len(unique),
                    "n_boot": n_boot, "reference_F1": float(reference_f1),
                    "comparison_F1": float(other_f1), "delta_F1": float(reference_f1 - other_f1),
                    "CI95_low": float(np.quantile(differences, 0.025)),
                    "CI95_high": float(np.quantile(differences, 0.975))})
    return pd.DataFrame(out)


def suite_variants(name: str, *, windows: Iterable[int] = WINDOWS,
                   cadences: Iterable[int] = CADENCES) -> list[Variant]:
    ref = Variant()
    ablations = [
        replace(ref, name="no_initial_sss", known_initial=False),
        replace(ref, name="raw_geometry", obs="raw"),
        replace(ref, name="linear_transition", transition="linear"),
        replace(ref, name="persistence_transition", transition="persistence"),
        replace(ref, name="transition_without_perclos", use_perclos=False),
        replace(ref, name="observation_only", fusion="observation_only"),
        replace(ref, name="transition_only", fusion="transition_only", obs="none"),
        replace(ref, name="session_only_calibration", cal="session_only"),
        replace(ref, name="time_only_calibration", cal="time_only"),
        replace(ref, name="latent_only_calibration", cal="latent_only"),
        replace(ref, name="observed_only_supervision", interpolated_training=False),
        replace(ref, name="context_only", model_type="context_only"),
        replace(ref, name="static_geometry_only", model_type="static_geometry"),
        replace(ref, name="initial_state_temporal_context_no_ocular", obs="none",
                fusion="transition_only", use_perclos=False),
    ]
    online = [
        replace(ref, name="online_both_048", feedback="both"),
        replace(ref, name="online_state_only_048", feedback="state"),
        replace(ref, name="online_intercept_only_048", feedback="intercept"),
        replace(ref, name="online_both_first2", feedback="both", feedback_ordinals=(0, 4)),
        replace(ref, name="online_both_first1", feedback="both", feedback_ordinals=(0,)),
    ]
    threshold = [replace(ref, name=f"threshold_sss_ge_{x}", fatigue_threshold=float(x))
                 for x in (3, 5)]
    grid = [replace(ref, name=f"grid_w{w}_c{c}", window_s=int(w), cadence_s=int(c))
            for w in windows for c in cadences]
    all_ = {"reference": [ref], "ablation": [ref, *ablations], "online": [ref, *online],
            "threshold": [ref, *threshold], "grid": grid}
    if name == "all":
        all_specs = [ref, *ablations, *online, *threshold, *grid]
        unique = {v.name: v for v in all_specs}
        return list(unique.values())
    if name not in all_:
        raise ValueError(name)
    return all_[name]
