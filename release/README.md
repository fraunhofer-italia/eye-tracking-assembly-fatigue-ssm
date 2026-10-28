# Artifact release

Code is published at https://github.com/fraunhofer-italia/operator-fatigue-wearable-eye-tracking.
The dataset and six fitted models are hosted separately on Hugging Face; their
repository IDs, pinned revisions and checksums are in `config/artifacts.json`.
All standalone release materials use the MIT License.

The dataset, models and generated results are ignored by Git. To prepare updated
Hugging Face upload folders locally:

```bash
python scripts/prepare_hf_upload.py
```

Review the contents of `hf_upload/dataset/` and `hf_upload/model/` before uploading.
The six model files comprise five cross-validation fits and one final offline
fit on all participants. Cross-validation scores do not constitute an independent
test of the final fit. After an authorized upload, refresh pinned revisions:

```bash
python3.11 scripts/update_hf_revisions.py
```

This helper requires `huggingface_hub`; it is not needed for numerical reproduction.
See the main README for installation and evaluation commands.
