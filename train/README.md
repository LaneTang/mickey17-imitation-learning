# Runtime compatibility helpers

`bc_model.py` and `bc_targets.py` remain here because the current ACT
checkpoint loader imports them through `deployment/preflight.py`.
They are not the training entry point for the showcased ACT model.

Earlier training scripts are preserved in [archive/training](../archive/training/).
The final checkpoint's training settings and split are documented in
[the reproduction guide](../docs/REPRODUCE.md#training-provenance).
`pipeline_config.toml` still contains historical training defaults alongside
active runtime settings; its values have not been changed.
