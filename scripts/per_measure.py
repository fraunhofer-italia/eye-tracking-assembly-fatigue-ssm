#!/usr/bin/env python3
"""Contribution of the seven geometry measures to the reference SSM (paper Section 4.3).

Exploratory analysis, run on the same dataset, folds and settings as the reference SSM:

1. Leave-one-measure-out: each geometry measure is removed in turn from the Ridge
   observation model, all five folds are refitted, and pooled F1 is compared with the
   full reference model using the paired participant bootstrap.
2. Standardized Ridge coefficients of the full reference model across the five
   training folds.

Usage:
    python scripts/per_measure.py            # writes results/per_measure/per_measure.csv
    python scripts/per_measure.py --check    # also compares with expected/per_measure.csv

The audited modules in src/ are not modified. Removing a measure only changes the
feature list passed to the observation model, through a wrapper around
paper_ssm.feature_names that is active while this script runs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paper_ssm as ssm  # noqa: E402
from fatigue_study import load_chain  # noqa: E402

LABELS = {
    "geo_eyelid aperture left [mm]_mean_delta": "Left aperture mean",
    "geo_eyelid aperture left [mm]_std_logratio": "Left aperture SD",
    "geo_eyelid aperture right [mm]_mean_delta": "Right aperture mean",
    "geo_eyelid aperture right [mm]_std_logratio": "Right aperture SD",
    "geo_vergence_deg_mean_delta": "Vergence mean",
    "geo_vergence_deg_std_delta": "Vergence SD",
    "eye_pose_std_delta": "Eye-angle variability",
}
DROP = "drop::"
_feature_names = ssm.feature_names


def feature_names_without_dropped(variant):
    names = _feature_names(variant)
    if variant.name.startswith(DROP):
        names = [n for n in names if n != variant.name[len(DROP):]]
    return names


def run(dataset: Path) -> pd.DataFrame:
    ssm.feature_names = feature_names_without_dropped
    chain = load_chain(dataset)
    reference = ssm.Variant()
    variants = [reference] + [replace(reference, name=DROP + f) for f in ssm.NORMALIZED]
    predictions, coefficients = [], []
    for variant in variants:
        rows = []
        for fold, people in ssm.FOLDS.items():
            test_mask = chain.participant_id.isin(set(people))
            train = chain[~test_mask].reset_index(drop=True)
            test = chain[test_mask].reset_index(drop=True)
            output, core = ssm.train_predict(train, test, variant)
            rows.append(ssm.exact_predictions(output, variant, fold))
            if variant.name == reference.name:
                coef = core.obs_model.named_steps["regressor"].coef_
                coefficients.append(dict(zip(ssm.feature_names(variant), coef), fold=fold))
        predictions.extend(rows)
        s = ssm.scores(pd.concat(rows))
        label = LABELS.get(variant.name[len(DROP):], variant.name)
        print(f"{label:24s} F1={s['F1']:.3f} ROC-AUC={s['ROC_AUC']:.3f}", flush=True)

    allp = pd.concat(predictions)
    boot = ssm.bootstrap_paired_f1(allp, reference=reference.name, n_boot=10_000)
    summary = {name: ssm.scores(group) for name, group in allp.groupby("experiment")}
    coef = pd.DataFrame(coefficients)
    out = []
    for f in ssm.NORMALIZED:
        b = boot[boot.comparison == DROP + f].iloc[0]
        out.append({
            "measure": LABELS[f],
            "coef_mean": coef[f].mean(),
            "coef_sd": coef[f].std(ddof=1),
            "same_sign_folds": int(max((coef[f] > 0).sum(), (coef[f] < 0).sum())),
            "F1_without": b.comparison_F1,
            "delta_F1": b.delta_F1,
            "CI95_low": b.CI95_low,
            "CI95_high": b.CI95_high,
            "AUC_without": summary[DROP + f]["ROC_AUC"],
        })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/dataset/dataset_eyetracking_5min.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/per_measure")
    parser.add_argument("--check", action="store_true", help="Compare with expected/per_measure.csv")
    args = parser.parse_args()
    result = run(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "per_measure.csv", index=False)
    print("\ndelta_F1 = reference minus model without the measure (positive: the measure helps)")
    print(result.round(3).to_string(index=False))
    if args.check:
        expected = pd.read_csv(ROOT / "expected/per_measure.csv")
        pd.testing.assert_frame_equal(result, expected, check_exact=False, atol=1e-10, rtol=1e-10)
        print("Matches expected/per_measure.csv", flush=True)


if __name__ == "__main__":
    main()
