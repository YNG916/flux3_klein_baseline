# Working on FLUX MVWD

- Human entry is `./mvwd`; keep utilities in `tools/`, launchers in `slurm/`.
- The user runs long experiments. Do not launch GPU training/inference/evaluation/profiling or allocate/submit Slurm jobs unless explicitly authorized; CPU implementation/tests and readiness checks are allowed.
- One experiment owns one independent training trajectory and all resumes, predictions, seeds, evaluations and zero-shot references. Never silently overwrite historical artifacts.
- Config, source, upstream, data split, assets and feature identities are frozen. Changes require a new experiment, not relaxing resume checks.
- Do not modify dataset files, shared assets or sibling baselines. Keep all environments, model downloads and artifacts within the workspace.
- Generation and captioning must remain GT-free. Review/metrics read GT only after complete generation and record their own access receipt.
- Disclose actual supervision and conditioning: two BEVs plus exact camera/robot metadata; no local BEV, mesh rendering or geometric loss in this variant.
- Report real test coverage and distinguish CPU tiny-DiT tests from actual pretrained GPU measurements.

## Canonical World Diffusion method branch

- User requested the full method before model ablations. CPU small-tensor correctness tests are required and are not model comparison experiments.
- New method implementation lives in `cw_diffusion/`; human commands are `./mvwd cw ...`, artifacts are `artifacts/cw/`.
- The baseline conditioning/supervision disclosure above describes the historical baseline. CW adds privileged train/val semantic, instance, depth, object-state and BEV-height supervision; disclose this difference.
- GT labels/boxes/object states may reach data preparation and loss functions only. Inference accepts RGB BEV and authorized robot/camera states; do not use test instance/semantic/height/occupancy or scene object state as conditions.
- Preserve native Euler/CFG, zero reference/text timesteps, independent image seeds, separate CFG memory histories and explicit target token offsets.
- Do not claim best architecture, H100 fit, real GPU correctness or image-quality gains from CPU tiny-DiT tests. GPU validation and long experiments still require explicit authorization.
