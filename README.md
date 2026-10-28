# Eye Tracking for Assembly Fatigue: SSM

Code for *Estimating Operator Fatigue During Manual Assembly with a Recursive
State-Space Model of Wearable Eye-Tracking Signals*.

The offline model estimates self-reported sleepiness (Stanford Sleepiness Scale,
SSS ≥ 4) from 150-second ocular summaries at approximately 300-second intervals.
The study includes 22 participants, 44 sessions and 504 evaluation endpoints.

## Install and download

Use Python **3.11**. From a clone of this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_artifacts.py
```

[Dataset](https://huggingface.co/datasets/fhi/eye-tracking-assembly-fatigue-ssm)
and [trained models](https://huggingface.co/fhi/eye-tracking-assembly-fatigue-ssm)
are hosted on Hugging Face. Downloads use pinned revisions and verify checksums.
The HF repositories currently require access: set `HF_TOKEN` with permission to
download the artifacts.

## Reproduce the results

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fatigue-study reproduce --check-models
```

This trains the five participant-held-out folds and checks the reference,
context-only, no-ocular, observed-only-supervision and feedback analyses against
`expected/`. It also verifies the last-available-SSS persistence comparator and
paired participant-bootstrap intervals. Outputs go to `results/reproduce/`.

To replay the saved reference models and feedback/persistence comparison:

```bash
fatigue-study evaluate
```

Reference pooled F1 is **0.713**; feedback-assisted F1 is **0.817**, with similar
observed F1 (**0.819**) for SSS persistence. Recent subjective assessments were
informative. Comparisons use participant-bootstrap intervals across 22 people;
see [model and validation details](docs/model.md).

## Use the final offline model

```python
from paper_bundle import load_model
from paper_ssm import load_dataset, prepare_window, build_sss_chain

model = load_model("artifacts/model/reference_all_participants.joblib")
data = load_dataset("artifacts/dataset/dataset_eyetracking_5min.csv", windows=[150])
chain = build_sss_chain(prepare_window(data, 150), 300)
predictions = model.predict(chain, feedback="none")
```

The final model was fitted on all participants; this example is inference on the
study cohort, not a held-out evaluation. The five `reference_fold_N.joblib` files
reproduce cross-validation; they are not an ensemble. New-participant inputs
require the same feature definitions and initial subjective/ocular references.
This package starts from processed features and is separate from the 30-second
real-time prototype; raw recordings are not included.

[Input dictionary](docs/dataset.md) · [Further commands](docs/model.md) ·
[Citation](CITATION.cff) · [MIT License](LICENSE) · [Attribution](NOTICE.md)
