# Retired scripts

One-off experiment scripts whose instruments were folded into heinrich
proper (2026-07-09/10). The scripts stay for provenance — the committed
`docs/data/*.json` goldens were produced by these exact files — but the
living instrument is the CLI/MCP tool:

| script | folded into |
|---|---|
| experiment_trough_channels.py | `profile-cb-channel-context` |
| experiment_pc_information.py | `profile-cb-pc-information` |
| experiment_anisotropy_a.py | `profile-cb-stream-spectrum` |
| verify_knn_lift.py | `profile-cb-knn-lift` |
| experiment_anisotropy.py | `profile-anisotropy` |
| experiment_cross_model.py + experiment_attribution.py + experiment_between.py | `profile-commit-anatomy` |
| experiment_neurons.py | `profile-commit-neurons` |
| experiment_paraphrase.py | `profile-commit-paraphrase` |
| experiment_sink.py | `profile-attention-sink` |
| cf_mri_prep.py | `mri-prep-web` |
| decepticon_migration.py | subsumed: `profile-cb-channel-context` over a checkpoint list gives the same three-channel bpb trajectory |
| homing_study_v4.py | superseded by `homing` / `heinrich_homing_run` + paper/heinrich_method.tex |
| figure2_homing_band.py, neuron_collapse_gif.py | paper-figure generators, not instruments |
| mri_finish.sh | obsolete (April-era pipeline) |
