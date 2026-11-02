# Demonstration collection

These are the unchanged collection scripts from the downloaded NUC source
snapshot. `record_demo_wrist_cam.py` records 30-second episodes by default
under the original NUC `4th_patch_extra` path. It operates robot hardware;
review the host, camera and serial configuration before use.

`plot_episode_commands.py` inspects recorded demonstrations. The alternative
Mac recorder, which supports `--output-dir`, is preserved under
[`archive/collection/`](../archive/collection/). Neither variant was changed
during reorganization.
