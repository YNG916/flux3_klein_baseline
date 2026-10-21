# Working on FLUX MVWD

- Human entry is `./mvwd`; keep utilities in `tools/`, launchers in `slurm/`.
- The user runs long experiments. Do not launch GPU training/inference/evaluation/profiling or allocate/submit Slurm jobs unless explicitly authorized; CPU implementation/tests and readiness checks are allowed.
- One experiment owns one independent training trajectory and all resumes, predictions, seeds, evaluations and zero-shot references. Never silently overwrite historical artifacts.
- Config, source, upstream, data split, assets and feature identities are frozen. Changes require a new experiment, not relaxing resume checks.
- Do not modify dataset files, shared assets or sibling baselines. Keep all environments, model downloads and artifacts within the workspace.
- Generation and captioning must remain GT-free. Review/metrics read GT only after complete generation and record their own access receipt.
- Disclose actual supervision and conditioning: two BEVs plus exact camera/robot metadata; no local BEV, mesh rendering or geometric loss in this variant.
- Report real test coverage and distinguish CPU tiny-DiT tests from actual pretrained GPU measurements.
