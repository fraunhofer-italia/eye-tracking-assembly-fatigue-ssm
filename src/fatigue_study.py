"""Standalone reproduction commands for the offline analyses in the revised paper."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import joblib
import scipy
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import paper_ssm as ssm
from paper_bundle import fit_model, load_model, save_model

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("reference", "context_only", "observed_only_supervision", "online_both_048",
            "initial_state_temporal_context_no_ocular")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def compare(actual, expected, keys):
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    if set(a.columns) != set(b.columns):
        raise AssertionError("Result columns differ from the reference release")
    pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False,
                                  check_exact=False, atol=1e-12, rtol=1e-12)


def load_chain(dataset: Path):
    origin = read_json(ROOT / "provenance/package_origin.json")
    if digest(dataset) != origin["data_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the paper dataset")
    for filename, expected in origin["source_files"].items():
        if digest(Path(__file__).with_name(filename)) != expected:
            raise ValueError(f"Audited source changed: {filename}")
    folds = read_json(ROOT / "config/folds.json")
    if folds != {str(k): list(v) for k, v in ssm.FOLDS.items()}:
        raise ValueError("Fixed participant folds differ")
    frame = ssm.prepare_window(ssm.load_dataset(dataset, windows=[150]), 150)
    chain = ssm.build_sss_chain(frame, 300)
    counts = (chain.participant_id.nunique(), chain.groupby(ssm.KEYS).ngroups,
              int(chain.is_initial_state_anchor.sum()), int(chain.is_eval_sss_window.sum()),
              int(chain.is_support.sum()))
    if counts != (22, 44, 44, 504, 15) or len(chain) != 563:
        raise ValueError(f"Unexpected study sequence counts: {counts}")
    return chain


def held_out(chain, fold):
    people = ssm.FOLDS[fold]
    train = chain[~chain.participant_id.isin(people)].copy().reset_index(drop=True)
    test = chain[chain.participant_id.isin(people)].copy().reset_index(drop=True)
    if not set(train.participant_id).isdisjoint(test.participant_id):
        raise AssertionError("Training and test participants overlap")
    return train, test


def checked_model(directory: Path, fold: int):
    manifest = read_json(directory / "manifest.json")
    info = manifest["models"][str(fold)]
    path = directory / info["filename"]
    if digest(path) != info["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path.name}")
    model = load_model(path)
    if model.metadata != info["training"] or model.metadata["test_participants"] != list(ssm.FOLDS[fold]):
        raise ValueError("Model fold provenance does not match")
    train_people = sorted(set().union(*map(set, ssm.FOLDS.values())) - set(ssm.FOLDS[fold]))
    if model.metadata["train_participants"] != train_people:
        raise ValueError("Model training participants do not match the fold")
    return model


def record_output(output, name, fold, feedback="none"):
    return ssm.exact_predictions(output, replace(ssm.Variant(), name=name, feedback=feedback), fold)


def check_persistence(chain, predictions, out):
    """Compare feedback to carrying the last disclosed SSS forward, before scoring."""
    records = []
    disclosures = 0
    for (participant, session), group in chain.groupby(ssm.KEYS, sort=False):
        last = float(group.loc[group.is_initial_state_anchor, "real_SSS_score"].iloc[0])
        for _, row in group.loc[group.is_eval_sss_window].iterrows():
            records.append({
                "participant_id": str(participant), "session": session,
                "prediction_time": row.prediction_time,
                "y_true": int(row.real_SSS_score >= 4), "y_pred": int(last >= 4),
                "last_available_sss": last,
                "experiment": "sss_persistence_feedback", "threshold_sss": 4.0,
            })
            if int(row.eval_ordinal) in (0, 4, 8):
                last = float(row.real_SSS_score)
                disclosures += 1
    carry = pd.DataFrame(records)
    feedback = predictions[predictions.experiment.eq("online_both_048")]
    matched = feedback.merge(carry, on=["participant_id", "session", "prediction_time"],
                             validate="one_to_one", suffixes=("_model", "_carry"))
    assert len(matched) == len(carry) == len(feedback) == 504
    np.testing.assert_array_equal(matched.y_true_model, matched.y_true_carry)
    paired = ssm.bootstrap_paired_f1(pd.concat([feedback, carry]),
                                    reference="online_both_048", n_boot=10000).iloc[0].to_dict()
    report = {
        "matched_evaluation_endpoints": len(carry), "disclosures": disclosures,
        "persistence_pooled_F1": float(f1_score(carry.y_true, carry.y_pred)),
        "persistence_accuracy": float(accuracy_score(carry.y_true, carry.y_pred)),
        "persistence_ROC_AUC": float(roc_auc_score(carry.y_true, carry.last_available_sss)),
        "feedback_model_pooled_F1": float(f1_score(feedback.y_true, feedback.y_pred)),
        "feedback_model_accuracy": float(accuracy_score(feedback.y_true, feedback.y_pred)),
        "feedback_model_ROC_AUC": float(roc_auc_score(feedback.y_true, feedback.probability)),
        "feedback_model_vs_persistence": paired,
        "ranking_score": "Last available continuous/ordinal SSS; no calibrated probability or Brier score assigned",
    }
    expected = read_json(ROOT / "expected/comparator_audit.json")
    for key, value in report.items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=1e-12)
    for key, value in paired.items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(value, expected["feedback_model_vs_persistence"][key],
                                       atol=1e-12, rtol=1e-12)
        else:
            assert value == expected["feedback_model_vs_persistence"][key]
    (out / "persistence_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def finish(chain, predictions, out: Path, *, mode: str, save_predictions: bool, model_checks: bool):
    predictions = pd.concat(predictions, ignore_index=True)
    persistence = check_persistence(chain, predictions, out)
    names = list(predictions.experiment.unique())
    fold_metrics, summary = ssm.summary_tables(predictions)
    for actual, filename, keys in [(fold_metrics, "fold_metrics.csv", ["experiment", "fold"]),
                                   (summary, "pooled_summary.csv", ["experiment"])]:
        expected = pd.read_csv(ROOT / "expected" / filename)
        compare(actual, expected[expected.experiment.isin(names)], keys)
        actual.to_csv(out / filename, index=False)
    bootstrap = ssm.bootstrap_paired_f1(predictions, reference="reference", n_boot=10000)
    expected = pd.read_csv(ROOT / "expected/paired_participant_bootstrap_f1.csv")
    compare(bootstrap, expected[expected.comparison.isin(names)], ["reference", "comparison"])
    bootstrap.to_csv(out / "paired_participant_bootstrap_f1.csv", index=False)
    reference = predictions[predictions.experiment.eq("reference")]
    prevalence = float(reference.y_true.mean())
    always_positive = 2 * prevalence / (1 + prevalence)
    np.testing.assert_allclose(always_positive, read_json(ROOT / "expected/always_positive_baseline.json")["F1"], atol=1e-12)
    if save_predictions:
        predictions.to_csv(out / "oof_predictions.csv", index=False)
    report = {
        "status": "passed", "command": mode, "analyses": names,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "rows_per_analysis": {name: int((predictions.experiment == name).sum()) for name in names},
        "aggregate_metrics_and_bootstrap_match": True, "comparison_atol_rtol": 1e-12,
        "fresh_vs_saved_model_predictions_match": model_checks,
        "always_positive_F1": always_positive,
        "persistence_comparator_verified": True,
        "persistence_pooled_F1": persistence["persistence_pooled_F1"],
        "reference_pooled_F1": float(summary.set_index("experiment").loc["reference", "F1"]),
        "feedback_pooled_F1": float(summary.set_index("experiment").loc["online_both_048", "F1"]),
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary[["experiment", "fold_F1_mean", "fold_F1_sd", "F1", "ROC_AUC", "AP", "Brier"]].to_string(index=False))
    print(f"Verification passed. Results: {out}", flush=True)


def evaluate(args):
    chain = load_chain(args.dataset)
    predictions = []
    for fold in ssm.FOLDS:
        _, test = held_out(chain, fold)
        model = checked_model(args.models, fold)
        for policy, name in [("none", "reference"), ("both", "online_both_048")]:
            predictions.append(record_output(model.predict(test, feedback=policy), name, fold, policy))
        print(f"Evaluated saved fold {fold}", flush=True)
    finish(chain, predictions, args.output, mode="evaluate", save_predictions=args.save_predictions, model_checks=False)


def reproduce(args):
    chain = load_chain(args.dataset)
    predictions = []
    for fold in ssm.FOLDS:
        train, test = held_out(chain, fold)
        model = fit_model(train)
        saved = checked_model(args.models, fold) if args.check_models else None
        for policy, name in [("none", "reference"), ("both", "online_both_048")]:
            output = model.predict(test, feedback=policy)
            if saved is not None:
                pd.testing.assert_frame_equal(output, saved.predict(test, feedback=policy),
                                              check_exact=False, atol=1e-12, rtol=1e-12)
            predictions.append(record_output(output, name, fold, policy))
        for name, changes in [("context_only", {"model_type": "context_only"}),
                              ("observed_only_supervision", {"interpolated_training": False}),
                              ("initial_state_temporal_context_no_ocular",
                               {"obs": "none", "fusion": "transition_only", "use_perclos": False})]:
            variant = replace(ssm.Variant(), name=name, **changes)
            output, _ = ssm.train_predict(train, test, variant, n_trees=200)
            predictions.append(ssm.exact_predictions(output, variant, fold))
        print(f"Reproduced five analyses and persistence comparator, fold {fold}", flush=True)
    finish(chain, predictions, args.output, mode="reproduce", save_predictions=args.save_predictions,
           model_checks=args.check_models)


def export(args):
    chain = load_chain(args.dataset)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty --output directory for model export")
    args.output.mkdir(parents=True, exist_ok=True)
    original = read_json(ROOT / "config/model_manifest.json")
    manifest = {k: v for k, v in original.items() if k != "models"}
    manifest["models"] = {}
    manifest["export_tool"] = {"name": "fatigue-study export", "sha256": digest(Path(__file__))}
    predictions = []
    for fold in ssm.FOLDS:
        train, test = held_out(chain, fold)
        metadata = original["models"][str(fold)]["training"]
        model = fit_model(train, metadata=metadata)
        path = save_model(model, args.output / f"reference_fold_{fold}.joblib")
        loaded = load_model(path)
        for policy in ("none", "both"):
            before = model.predict(test, feedback=policy)
            pd.testing.assert_frame_equal(before, loaded.predict(test, feedback=policy), check_exact=True)
            name = "reference" if policy == "none" else "online_both_048"
            predictions.append(record_output(before, name, fold, policy))
        manifest["models"][str(fold)] = {
            "filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
            "roundtrip_exact": True, "training": metadata,
        }
        print(f"Exported and verified fold {fold}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finish(chain, predictions, args.output, mode="export", save_predictions=False, model_checks=False)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "evaluate",
                    "--dataset", str(args.dataset.resolve()), "--models", str(args.output.resolve()),
                    "--output", str((args.output / "fresh_process_check").resolve())], check=True)
    print(f"Complete bundles: {args.output}", flush=True)


FINAL_NAME = "reference_all_participants"
PREDICTION_FIELDS = ("latent", "latent_variance", "probability", "y_pred_sss")


def fingerprints(output):
    return {column: hashlib.sha256(np.ascontiguousarray(
        output[column].to_numpy(dtype="<f8")).tobytes()).hexdigest()
        for column in PREDICTION_FIELDS}


def fit_final(args):
    """Fit one inference model on all participants; do not estimate test scores."""
    chain = load_chain(args.dataset)
    out = args.output
    names = [f"{FINAL_NAME}{suffix}" for suffix in (".joblib", ".manifest.json", ".verification.json")]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Final model already exists here; select a new --output directory")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_role": "final_offline_inference_all_participants",
        "train_participants": sorted(chain.participant_id.unique()), "test_participants": [],
        "training_sequence_rows": len(chain),
        "observed_training_endpoints": int(chain.is_eval_sss_window.sum()),
        "initialization_references": int(chain.is_initial_state_anchor.sum()),
        "support_points": int(chain.is_support.sum()),
        "dataset_sha256": digest(args.dataset),
        "model_source_sha256": digest(Path(ssm.__file__)),
    }
    model = fit_model(chain, metadata=metadata)
    original = {policy: model.predict(chain, feedback=policy) for policy in ("none", "both")}
    path = save_model(model, out / names[0])
    loaded = load_model(path)
    for policy, before in original.items():
        pd.testing.assert_frame_equal(before, loaded.predict(chain, feedback=policy), check_exact=True)
    template = read_json(ROOT / "config/model_manifest.json")
    manifest = {
        "format": template["format"], "model_role": metadata["model_role"],
        "artifact": {"filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size},
        "training": metadata, "configuration": template["configuration"],
        "feature_order": template["feature_order"], "fixed_settings": template["fixed_settings"],
        "python": platform.python_version(),
        "libraries": {m.__name__: m.__version__ for m in (np, pd, scipy, sklearn, joblib)},
        "sources_sha256": {p.name: digest(p) for p in
                           (Path(__file__), Path(ssm.__file__), Path(__file__).with_name("paper_bundle.py"))},
        "learned_parameters": {"target_mean": model.core.mean, "target_std": model.core.std,
                               "Q": model.core.Q, "R": model.core.R,
                               "observation_training_points": model.core.n_obs_train,
                               "transition_training_pairs": model.core.n_transition_pairs},
        "serialization_roundtrip_exact": True,
        "prediction_fingerprints": {policy: fingerprints(output) for policy, output in original.items()},
        "independent_test_performance": None,
        "performance_note": "Fit on all 22 participants. Paper cross-validation scores belong to the five held-out fits; no independent test score is assigned to this final fit.",
    }
    (out / names[1]).write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify-final",
                    "--dataset", str(args.dataset.resolve()), "--models", str(out.resolve())], check=True)
    print(f"Final offline model trained on all 22 participants: {path}", flush=True)


def verify_final(args):
    """Verify final-fit provenance, fresh-process loading and blind inference."""
    chain = load_chain(args.dataset)
    manifest = read_json(args.models / f"{FINAL_NAME}.manifest.json")
    path = args.models / manifest["artifact"]["filename"]
    if digest(path) != manifest["artifact"]["sha256"]:
        raise ValueError("Final model checksum mismatch")
    model = load_model(path)
    metadata = model.metadata
    if metadata != manifest["training"] or metadata["model_role"] != "final_offline_inference_all_participants":
        raise ValueError("Final model role/provenance mismatch")
    if metadata["train_participants"] != sorted(chain.participant_id.unique()) or metadata["test_participants"]:
        raise ValueError("Final model must use all participants and have no held-out cohort")
    if metadata["training_sequence_rows"] != len(chain) or metadata["dataset_sha256"] != digest(args.dataset):
        raise ValueError("Final model training data mismatch")
    for policy in ("none", "both"):
        if fingerprints(model.predict(chain, feedback=policy)) != manifest["prediction_fingerprints"][policy]:
            raise AssertionError("Fresh-process predictions differ from the pre-export model")
    blinded = chain.copy()
    blinded.loc[~blinded.is_initial_state_anchor, "real_SSS_score"] = np.nan
    with_labels = model.predict(chain)
    without_labels = model.predict(blinded)
    for column in PREDICTION_FIELDS:
        np.testing.assert_array_equal(with_labels[column], without_labels[column])
    report = {
        "status": "passed", "model_role": metadata["model_role"],
        "training_participants": 22, "training_sessions": 44, "sequence_points": len(chain),
        "initializations": 44, "observed_endpoints": 504, "support_points": 15,
        "serialization_roundtrip_exact": manifest["serialization_roundtrip_exact"],
        "fresh_process_predictions_identical": True,
        "blind_inference_unchanged_without_noninitial_sss": True,
        "independent_test_performance": None,
    }
    (args.models / f"{FINAL_NAME}.verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "reproduce", "export", "fit-final", "verify-final"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/dataset/dataset_eyetracking_5min.csv")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-models", action="store_true", help="Compare freshly fitted states/probabilities to saved models")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "artifacts/model" if args.command == "fit-final" else ROOT / "results" / args.command
    if args.command != "export":
        args.output.mkdir(parents=True, exist_ok=True)
    {"evaluate": evaluate, "reproduce": reproduce, "export": export,
     "fit-final": fit_final, "verify-final": verify_final}[args.command](args)


if __name__ == "__main__":
    main()
