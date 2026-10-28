---
license: mit
pretty_name: "Eye Tracking for Assembly Fatigue: SSM — Processed Features"
tags:
  - wearable-eye-tracking
  - manual-assembly
  - sleepiness
  - tabular
  - timeseries
---

# Eye Tracking for Assembly Fatigue: SSM — Dataset

Dataset accompanying *Estimating Operator Fatigue During Manual Assembly with a
Recursive State-Space Model of Wearable Eye-Tracking Signals*.

## Contents

The original extracted table contains 15,749 candidate rows and 742 columns,
from 22 participants and 44 morning/afternoon sessions. The reported reference
analysis uses 150-s ocular windows and a nominal 300-s assessment cadence,
retaining 44 initialization points, 504 evaluation endpoints and 15 support
points. There are 208 elevated-sleepiness outcomes (SSS ≥ 4).

See `DATA_DICTIONARY.md` for the 19 columns read by the model, units, reference
policies, exclusions and the participant-3 afternoon initialization fallback.
The dataset retains unused auxiliary columns to preserve the original artifact.

## Reproduction

Use the companion `eye-tracking-assembly-fatigue-ssm` code package and
its recorded dependencies, folds and preprocessing conventions. The source CSV
SHA-256 is
`ff5f537bc54f9b1ba2c031594acc3c8cdb792b9601c5e267e42dd664f3ced7c0`.

The [model repository](https://huggingface.co/fhi/eye-tracking-assembly-fatigue-ssm)
contains the fitted models and Python modules needed for loading them. The full
code is available in the [GitHub repository](https://github.com/fraunhofer-italia/eye-tracking-assembly-fatigue-ssm).

## Scope

The outcome is repeated self-reported Stanford Sleepiness Scale sleepiness,
used as a fatigue-related state. This table supports offline reproduction from
extracted features; original Neon recordings and questionnaire sources are not
included. It does not independently validate a complete mental-fatigue measure,
continuous deployment or adaptive robot assistance.

## Distribution and citation

This extracted feature dataset is distributed under the MIT License; see
`LICENSE` and `NOTICE.md` for the license text and attribution. Cite the
accompanying paper and the specific dataset revision used. Publication
identifiers will be added when available.
