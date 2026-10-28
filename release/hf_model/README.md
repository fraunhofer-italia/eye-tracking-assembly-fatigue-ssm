---
license: mit
tags:
  - wearable-eye-tracking
  - manual-assembly
  - sleepiness
  - state-space-model
  - scikit-learn
---

# Eye Tracking for Assembly Fatigue: SSM

Complete fitted models accompanying *Estimating Operator Fatigue During Manual
Assembly with a Recursive State-Space Model of Wearable Eye-Tracking Signals*.

## Files and intended use

**Final model for inference:** `reference_all_participants.joblib`, trained on
all 22 participants and 44 sessions with the paper's 150 s / 300 s reference
configuration. Its matching `.manifest.json` and `.verification.json` identify
the full-data fit and record serialization and fresh-process checks. No
independent test-performance score is assigned to this final fit.

**Models for numerical reproduction:**
`reference_fold_1.joblib` through `reference_fold_5.joblib` reproduce five fixed
participant-held-out folds. Each contains fitted Ridge, Extra Trees and logistic
pipelines, imputation/scaling, Q/R, target mean/SD and training/test provenance.
`manifest.json` records hashes, configuration, feature ordering and library
versions. Those five fold files are not an ensemble; the all-participant fit is
provided separately as the primary inference artifact.

The configuration is **150-s trailing ocular features / 300-s nominal state
cadence**. It is separate from the 150-s/30-s realtime prototype.

## Loading

The repository includes `paper_ssm.py`, `paper_bundle.py` and `requirements.txt`.
After downloading its files, install the pinned dependencies with
`python3.11 -m pip install -r requirements.txt` and run Python from that directory:

```python
from paper_bundle import load_model
model = load_model("reference_all_participants.joblib")
output = model.predict(prepared_chain, feedback="none")
```

`prepared_chain` is constructed from the extracted feature CSV with
`load_dataset(..., windows=[150])`, `prepare_window(..., 150)`, and
`build_sss_chain(..., 300)`. It includes initial subjective/ocular references.
For cross-validation reproduction, load a `reference_fold_N.joblib` file and use
the matching held-out participant subset. These custom joblib
bundles require `paper_ssm.py` and `paper_bundle.py` from the companion package;
they are not standalone executables. Load only trusted serialized files.

The same bundle supports `feedback="both"`: reveal the first, fifth and ninth
eligible session ratings after their predictions, update state and session logit
offset, and keep all fitted population parameters fixed.

The [dataset repository](https://huggingface.co/datasets/fhi/eye-tracking-assembly-fatigue-ssm)
contains the extracted input features. The code is available on [GitHub](https://github.com/fraunhofer-italia/eye-tracking-assembly-fatigue-ssm).

## Cross-validation performance

Across 504 held-out endpoints, reference pooled F1 is 0.713, ROC-AUC 0.787 and
Brier score 0.188. Selective post-prediction feedback gives F1 0.817, ROC-AUC
0.894 and Brier score 0.123. These scores belong to the five held-out fits, not
an independent test of the final all-participant model.
Carrying forward the last available SSS with the same feedback schedule gives
F1 0.819, accuracy 0.853 and ROC-AUC 0.919. The feedback-assisted model and this
comparator achieved similar observed pooled F1 (paired difference -0.002,
95% interval [-0.039, 0.033]), highlighting the predictive value of recent
subjective assessments.
Serialization/reloading was checked for exact
prediction equality, including loading in a fresh Python process.

The outcome is SSS-defined self-reported sleepiness. The reference model had
higher observed pooled F1 than the contextual controls, with paired intervals
including zero. The 504 assessments come from 22 participant resampling units,
which limits precision for modest performance differences. The seven geometry
inputs describe eyelid opening, binocular alignment and ocular variability.
The paper documents their literature rationale and the exploratory temporal
comparison supporting the 150-s/300-s configuration. Larger independent cohorts
and prospective evaluation can assess use in workplace monitoring.

## Distribution and citation

The trained model artifacts and accompanying code are distributed under the MIT
License; see `LICENSE` and `NOTICE.md` for the license text and attribution. Cite
the paper and specific model revision used. Publication identifiers will be
added when available.
