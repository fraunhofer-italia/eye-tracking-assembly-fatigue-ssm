# Verification records

`manuscript_alignment.json` records a historical numerical audit against the
manuscript sources identified by its hashes. `original_analysis.json` records
the fitting run and environment; neither is a checksum of the latest prose.

`standalone_verification.json` and `final_model_verification.json` record the
standalone evaluation and final-fit checks. The current editorial alignment is
recorded separately in `documentation_alignment.json`. Historical numerical
results and audited source hashes are preserved.

Raw study recordings are managed outside this feature/model release. The
numerical audits start from the extracted table; preprocessing is implemented
in the study toolkit and described in the manuscript.
