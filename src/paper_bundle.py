"""Complete fitted bundles for the paper's offline 150 s / 300 s model.

Uses the unchanged paper_ssm fitting/inference routines. A bundle contains the
population estimator and classifier; participant-session state is initialized
anew when predict() processes a chronological prepared chain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import paper_ssm as ssm

FORMAT = "fatigue-paper-offline-150s-300s-v1"


@dataclass
class PaperModel:
    core: ssm.Core
    calibrator: Pipeline
    metadata: dict

    def predict(self, chain: pd.DataFrame, *, feedback: str = "none") -> pd.DataFrame:
        """Predict a chain from load_dataset -> prepare_window -> build_sss_chain.

        feedback='both' reproduces the secondary first/fifth/ninth disclosure
        analysis, using the same fitted population parameters as feedback='none'.
        """
        if feedback not in ("none", "state", "intercept", "both"):
            raise ValueError(f"Unknown feedback policy: {feedback}")
        variant = self.core.variant
        output = ssm.infer_sequence(
            chain, self.core, self.calibrator, feedback=feedback,
            feedback_ordinals=variant.feedback_ordinals, threshold=variant.fatigue_threshold,
        )
        output["training_target_mean"] = self.core.mean
        output["training_target_std"] = self.core.std
        return output


def fit_model(train: pd.DataFrame, *, metadata: dict | None = None) -> PaperModel:
    """Fit only supplied training participants, following train_predict exactly.

    The small classifier-fitting block is kept explicit here so the audited
    paper_ssm.py and the archived analysis source hashes remain unchanged.
    Export verification compares every held-out output field to that analysis.
    """
    variant = ssm.Variant()
    core = ssm.fit_core(train, variant, n_trees=200)
    train_latents = ssm.infer_sequence(train, core).latent.to_numpy()
    observed = train.is_eval_sss_window.to_numpy(dtype=bool) & train.real_SSS_score.notna().to_numpy()
    calibrator = ssm.calibrator_pipeline()
    inputs = ssm.calibration_inputs(train.loc[observed], train_latents[observed], variant.cal)
    labels = (train.loc[observed, "real_SSS_score"] >= variant.fatigue_threshold).astype(int)
    calibrator.fit(inputs, labels)
    # Store sklearn estimators, without the optional numba acceleration cache.
    return PaperModel(replace(core, packed_transition=None), calibrator, dict(metadata or {}))


def save_model(model: PaperModel, path: str | Path) -> Path:
    """Atomically save pipelines, classifier, Q/R, target scaling and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump({"format": FORMAT, "model": model}, temporary, compress=3, protocol=5)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_model(path: str | Path) -> PaperModel:
    """Load a trusted offline paper bundle, not a realtime deployment bundle."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Not an offline paper-model bundle")
    model = payload.get("model")
    if not isinstance(model, PaperModel) or model.core.variant != ssm.Variant():
        raise ValueError("Incompatible paper-model configuration")
    return model
