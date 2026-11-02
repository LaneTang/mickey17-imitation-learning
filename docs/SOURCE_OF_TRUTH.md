# Source provenance and repository scope

## NUC source snapshot

The downloaded `Mickey17 remote/Code/IL` directory is the authoritative **downloaded source snapshot**. It was compared against the earlier Mac project on 2026-10-30: all 28 eligible files matched after synchronization. This does not verify the current remote NUC filesystem.

`source_provenance.json` preserves the original SHA-256 hashes of the 25 included NUC files. Two old backup scripts and the old transfer manifest were omitted. Original source hashes have not been rewritten to describe subsequent edits. The existing `deployment/rollout_act_overlap.py` includes a later local Space-key timing/label extension, recorded separately in `local_extensions.json`. Its control/scheduling behavior has not been changed by this showcase update.

Run:

```bash
python3 tools/verify_source_snapshot.py
```

The verifier reports unchanged NUC files and separately verifies explicitly recorded local extensions. An unexplained mismatch fails. A local-extension hash is not represented as a NUC match.

## Local work and archives

The archival reorganization is recorded in `source_relocations.json`.
Original NUC hashes are unchanged; the verifier resolves their original paths
through this separate map. Every moved file retains its original bytes.
Historical rollout/training alternatives are under `archive/`; the NUC
collection scripts are now under `collection/`. The final runtime and all
its current import dependencies remain at their original paths.

The Mac project supplied collection/training wrappers, review code, assets and documentation beyond the deployment-only NUC download. Matching source paths used the NUC version during synchronization. The five overwritten Mac versions remain in the original workspace's `analysis/2026-10-30/mac-before-nuc-sync/`, outside this repository; original workspace files were not deleted.

The author's starting README and experiment statement are archived verbatim under [docs/archive](archive/README.md), with their starting hashes. The reconciled experiment statement records the author's clarified 0–5 rubric, including rank 4 = placed but not returned. The rollout's historical prompt is documented separately and has not been silently changed.

The current showcase adds a portable evidence importer/analyzer, tests, scientific figures, selected derived trajectories, compressed illustrative media and supporting documentation. These are local extensions, not part of the original NUC source download. Original media are preserved unchanged locally; metadata-stripped/compressed copies are published instead. Two previously tracked source videos are removed from the current Git tree, without deleting local originals or rewriting existing history. Historical analysis is preserved.

## Evidence provenance

The November 1 archive's SHA-256 is `f9bec52554f3706d93810b7e9e6fade4a4783c83b54151df8f5f4fb58c3ad719`. It was verified before a separate extraction. Current [evidence](../evidence/2026-11-01/README.md) and [analysis](../analysis/2026-11-01) describe 95 primary trials and twelve supplementary attempts, deduplicated to 107 unique IDs. Source hashes refer to original files; raw logs, labels, camera images, checkpoints and archives were not edited or moved.

[analysis/2026-10-25](../analysis/2026-10-25) preserves the earlier 70-trial study and its original nonportable script. Its different scoring is not pooled with current results. The portable entry point is `tools/analyze_outcomes.py`; it defaults to public evidence and supports an optional extracted raw directory.

## External dependencies

This repository is project code, not a complete LeRobot fork. It requires an installed LeRobot environment for training and hardware inference. The Mac checkout's base commit is `c364df18c54b670b6450073039aa2cd6024a6855`, with additional local changes outside this directory. The NUC's installed revision/environment was not captured. That Mac commit alone does not reproduce deployment.

`support/configuration_act.py` is a reference ACT configuration extension, not automatically installed. Historical setup and command documents retain old paths. Empirical limits and default poses are specific recorded configurations, not universal calibration or collision detection.

The [reproduction guide](REPRODUCE.md) separates verified offline analysis from unavailable training data/weights and machine-specific deployment. No robot operation, dependency installation on the NUC or training was performed for this update. Upstream notices and the original license copy are preserved; no project-wide license was assigned.

The showcase is published at [LaneTang/mickey17-imitation-learning](https://github.com/LaneTang/mickey17-imitation-learning).
