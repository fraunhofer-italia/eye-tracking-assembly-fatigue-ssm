# Dataset and input dictionary

File: `dataset_eyetracking_5min.csv`, 172,474,471 bytes, **15,749 rows and 742
columns**. SHA-256:
`ff5f537bc54f9b1ba2c031594acc3c8cdb792b9601c5e267e42dd664f3ced7c0`.
The original extracted table is preserved without column/row reduction.

Rows are candidate causal feature summaries for 22 participants and two sessions
per participant. The reference recursive state-space model (SSM) reads only
**19 columns** at the 150-s
scale; the remaining feature scales, diagnostics and annotations are
not predictors of the reported reference SSM.

## Metadata used by the loader

| Column | Meaning |
| --- | --- |
| `participant_id` | Pseudonymous study identifier, normalized to strings `1`–`22` |
| `session` | `morning` or `afternoon` |
| `task` | Task label; assembly prediction candidates use `assembly_realtime`, a legacy label that does not make this analysis realtime |
| `prediction_time` | Candidate endpoint in seconds in the recording's time base; at an assembly assessment, the annotated assembly-block boundary associated with its SSS rating |
| `assembly_start_time` | Assembly start in the same time base; elapsed minutes are max(0, endpoint minus start)/60 |
| `SSS_score` | Legacy dense/interpolated value; read for compatibility but **not used** as the model's supervision |
| `real_SSS_score` | Observed SSS at evaluation endpoints, or the starting subjective reference at an initialization point; other values may be missing |
| `is_eval_sss_window` | True at exact eligible observed assembly assessment endpoints |
| `is_initial_state_anchor` | True at the single starting reference for each participant-session |
| `exclude_from_training_eval` | Exclusion flag for assembly candidates |

Boolean fields accept true/false or equivalent 1/0 indicators. Empty numeric
values are read as missing. Continuous training targets are reconstructed from
`real_SSS_score` only within training participants; the archived `SSS_score`
interpolation is never supplied as a held-out target.

## Nine columns for the 150-s scale

| Column | Definition / units |
| --- | --- |
| `feature_window_valid_w150s` | Validity flag used to retain candidate windows |
| `perclos_mean_w150s_within` | Dimensionless relative continuous eyelid closure: normalized closure minus the morning-reference closure |
| `geo_eyelid aperture left [mm]_mean_w150s` | Mean left aperture, mm |
| `geo_eyelid aperture left [mm]_std_w150s` | Sample SD of left aperture, mm |
| `geo_eyelid aperture right [mm]_mean_w150s` | Mean right aperture, mm |
| `geo_eyelid aperture right [mm]_std_w150s` | Sample SD of right aperture, mm |
| `geo_vergence_deg_mean_w150s` | Mean angle between optical axes, degrees |
| `geo_vergence_deg_std_w150s` | Sample SD of that angle, degrees |
| `eye_pose_std_w150s` | Mean sample SD of three angles derived from the left optical axis, degrees; not independently measured ocular torsion |

The closure variable is a continuous aperture-based index, not the conventional
percentage of time beyond an 80%-closure threshold. It uses fixed eye-specific
95th-percentile morning aperture references and morning closure subtraction in
both sessions. Positive relative values mean greater closure than at the morning
reference; subtraction can produce negative relative values.

Five geometry inputs are differenced from the feature vector at that session's
initialization point. Left/right aperture SDs use log((current+0.001)/(reference+0.001)).
Training medians and scaling statistics are fitted within each fold.

## Starting references, labels and sequence selection

- Morning SSS starts from the baseline rating.
- In 21 afternoon sessions, the starting value is the mean of up to six available
  change-detection ratings (`SSS_13`–`SSS_18`), so it can be fractional.
- Participant 3 afternoon starts at the first assembly SSS (`SSS_19`), which is
  excluded from evaluation. There are 43 pre-assembly starts plus this fallback.
- SSS 1–3 is the lower class; SSS 4–7 is the elevated class.
- Valid retained sequences contain 44 initializations, 504 evaluation points and
  15 intermediate support points: 563 points overall. Preserve the candidate
  table for selection; publishing only the 504 evaluation rows is insufficient.
- Intermediate requests are selected retrospectively using assessment times and
  a nominal 300-s cadence. Only training participants receive interpolated
  continuous targets; test support states use their previous predicted state.
- All 504 evaluation endpoints have complete 150-s intervals and all eight
  finite ocular predictors. Seventeen initialization windows are shorter.

This release starts from the extracted feature table. Raw Neon recordings and
questionnaire sources are managed separately, and the study toolkit implements
the upstream preprocessing pipeline. The manuscript specifies 10-Hz time bins,
limited within-window interpolation and signal-coverage checks. Auxiliary
columns preserve the original extracted table; the reference SSM uses the
19-column input contract described above.

## Endpoint and coordinate conventions

At SSS assessment endpoints, feature windows end at the annotated assembly-block
boundary. For paired questionnaire markers this is the first marker of the
interruption; singleton markers are used
as available boundaries, and terminal blocks can end at assembly end. Feature
windows align to these annotated event boundaries.

Optical-axis components use the native Neon scene-camera coordinate frame,
without rotation into world or anatomical coordinates. See the
[Neon data-stream documentation](https://docs.pupil-labs.com/neon/data-collection/data-streams/).
