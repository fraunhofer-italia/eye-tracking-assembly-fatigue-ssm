# Model and evaluation

The estimator predicts SSS-defined sleepiness during manual assembly. It is a
population-trained offline model using participant-specific initial ocular
and subjective references. It provides a reproducible basis for further
evaluation of sleepiness monitoring during manual work.

## Fitted components

1. **Current ocular estimate:** median imputation, standard scaling and Ridge
   regression (`alpha=1`) on seven session-reference-normalized geometry inputs.
2. **Temporal estimate:** Extra Trees regression on previous standardized state
   and current relative eyelid closure. It uses 200 trees, depth 4, minimum leaf
   size 5, random seed 42 and one worker. Inputs use training-median imputation.
3. **Recursive fusion:** Q/R are population variances of fitted training
   residuals plus 1e-4. At each update, P_prior=P_previous+Q,
   K=P_prior/(P_prior+R), state=temporal+K*(ocular-temporal),
   P=(1-K)*P_prior. Initial variance is 1e-4.
4. **Binary classifier:** training-median imputation, standard scaling and
   balanced logistic regression (`liblinear`, C=1, max_iter=1000, seed=42) on
   [state, afternoon, state*afternoon, elapsed_minutes, state*elapsed_minutes].
   Probability ≥ 0.5 predicts SSS ≥ 4.

Training targets use the mean and population SD of retained training targets,
including initialization and interpolated support points. The regressions use
non-initialization points. Transition fitting receives the preceding known
training target; inference receives the preceding estimated state. Filtering
the training sequences supplies the classifier's states; classifier fitting
uses only observed assembly endpoints and no additional inner cross-validation.

## Reported conditions

| Name | Description |
| --- | --- |
| `reference` | Complete model, subsequent test SSS withheld after initialization |
| `context_only` | Balanced logistic classifier on [afternoon, elapsed_minutes, afternoon*elapsed_minutes] |
| `observed_only_supervision` | Same reference architecture and sequence points, with continuous supervision restricted to finite observed starting references and ratings |
| `online_both_048` | Same fitted reference parameters, with state reset and probability-offset adaptation after selected ratings |
| `initial_state_temporal_context_no_ocular` | Initial SSS, learned autoregressive Extra Trees transition and contextual classifier; no ocular inputs or observation/fusion update |

Feedback is disclosed at the first, fifth and ninth eligible assessment of each
session, **after its own prediction**. A state reset uses the standardized
rating and variance 1e-4. A session-specific logit offset minimizes mean binary
cross-entropy plus 0.1*offset² over accumulated feedback, bounded to [-6,6].
The numerical optimizer uses at most 70 Newton updates, gradient curvature floor
1e-8 and change tolerance 1e-10. Probabilities are clipped to [1e-9,1-1e-9]
before conversion to logits. There are 128 disclosures across 44 sessions.
The global fitted pipelines remain fixed.

Both sessions of a participant remain in the same fold; see `config/folds.json`.
Mean fold F1 and sample SD describe between-fold variability. Comparisons and
participant-bootstrap inference use pooled F1. ROC-AUC, average precision and
Brier score also use all 504 held-out endpoints. Bootstrap intervals
use 10,000 paired participant resamples, retaining each selected participant's
complete held-out observations, with NumPy default_rng(42) and string-sorted IDs.
Bootstrap CSV differences are **reference minus comparator**; the feedback
improvement has the opposite sign to that row's stored difference.

## Saved bundles

`reference_fold_1.joblib` through `reference_fold_5.joblib` each contain:

- Fitted Ridge, Extra Trees and logistic pipelines, including imputation/scaling.
- Target mean/SD and residual variances Q/R.
- The complete reference Variant and participant-fold provenance.

The fold models have different training participants. They are provided to replay
cross-validation, not as an ensemble. One file supports both blind and feedback
inference, since feedback updates sequence state without changing global fitted
parameters. Personal sequence state is reconstructed when `predict()` processes
a chain; the file is not a live session checkpoint.

## Final all-participant inference model

`reference_all_participants.joblib` is fitted separately on all 22 participants,
44 sessions and 563 retained sequence points with the same reference settings.
It contains all fitted pipelines, target scaling and Q/R, and has no held-out
participants. Its role is identified explicitly in model metadata and in
`reference_all_participants.manifest.json`; the fold models remain available.

Use `fatigue-study fit-final` to generate it and `fatigue-study verify-final`
to check loading and blind-label independence. Serialization is checked both
before/after loading and in a fresh Python process, in blind and feedback modes.
These checks verify numerical integrity on the study inputs. Reported
performance comes from the five participant-held-out fits; external evaluation
of the final all-participant model is a future step.

Its update cadence remains 300 s. Inference takes a prepared offline sequence
with initial reference information. The 30-s realtime prototype is a separate
implementation with its own input and model contracts.

Import names `paper_ssm` and `paper_bundle` are kept stable for joblib loading.
The original `.joblib` hashes and configuration are recorded in
`config/model_manifest.json` and `config/artifacts.json`. Preserve these code
modules and the pinned scientific Python versions with the files.

## Persistence comparator

The comparator starts from the same initial session SSS and carries it forward,
replacing it only after the first, fifth and ninth assessment predictions have
been scored. The same 128 disclosures are used on the same 504 endpoints.
SSS ≥ 4 supplies its class; the last available ordinal SSS supplies its ROC-AUC
ranking score. It is not a calibrated probability, so no Brier score is assigned.

Persistence achieves pooled F1 0.819, accuracy 0.853 and ROC-AUC 0.919. For the
feedback-assisted model minus persistence, pooled ΔF1 is -0.002 with paired
participant-bootstrap 95% interval [-0.039, 0.033]. The two approaches had
similar observed pooled F1, highlighting the predictive value of recent
subjective assessments. The no-ocular control has F1 0.659; reference minus
control is 0.054 with interval [-0.050, 0.160]. Full-precision expectations are
included in `expected/`; each evaluation writes `persistence_comparison.json`.

## Configuration rationale and interpretation

The seven geometry measures compactly describe eyelid opening, binocular
alignment and ocular variability. The accompanying paper gives the literature
rationale for these measurement families. A prior exploratory comparison of
25 temporal configurations supported the 150-s window and 300-s update interval:
this combination had the highest observed pooled F1, with similar performance
at nearby settings. The predictor set and temporal settings were fixed during
the reported cross-validation. The earlier temporal comparison is separate
from a fresh grid run with this release.

The reference model achieved higher observed pooled F1 than both contextual
controls, with paired intervals including zero. Observed-only supervision
yielded F1 0.724, compared with 0.713 for the reference training procedure.
The 504 assessments are repeated measurements from 22 participants, who form
the independent resampling units. This cohort size limits precision for modest
performance differences; intervals spanning zero express uncertainty about
their magnitude and direction. Larger independent cohorts and prospective
evaluation can assess consistency across participants and settings.

The numerical checks in `provenance/` identify their audited source versions.
See `provenance/README.md` for historical snapshots and the current documentation
alignment record.

## Additional commands

```bash
fatigue-study reproduce --save-predictions
fatigue-study verify-final
fatigue-study export --output results/exported_models
fatigue-study fit-final --output results/final_refit
fatigue-study verify-final --models results/final_refit
python scripts/download_artifacts.py --check-only
```

Exports and refits require an empty destination. Individual predictions,
downloaded artifacts and generated results are ignored by Git.
