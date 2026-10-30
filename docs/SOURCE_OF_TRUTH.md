# Source provenance and repository scope

Prepared locally on 2026-10-30. The user-authored root README and Roadmap are preserved verbatim and remain drafts.

## Authoritative deployment source

The downloaded NUC directory `Mickey17 remote/Code/IL` is the source of truth for source/configuration files it contains. Its contents were compared by SHA-256 with the Mac `lerobot/Atom's/Imitation Learning` project before preparing this standalone repository. All 28 eligible files matched after synchronization. This verifies the downloaded snapshot, not the current remote NUC filesystem.

The standalone copy includes 25 of those files unchanged. Two historical `*.before*` scripts and the old `transfer_manifest.json` are deliberately omitted. That old manifest describes an earlier bundle and contains hashes that no longer represent every current source file. `source_provenance.json` records freshly calculated hashes instead.

The synchronization replaced five existing Mac files and added eight. Previous versions of the five replaced files remain in the original workspace at `analysis/2026-10-30/mac-before-nuc-sync/`, outside this repository. No source file was removed from the original workspace.

The copied empirical limits are exactly the NUC versions. In particular, their proposal-gap thresholds differ from the previous Mac copy. They are not physical joint limits or collision detection. Root and teleoperation default poses are copied reference configurations for the existing robot; they are not universal calibration values.

## Mac-only project material

Training wrappers, dataset preparation code, offline review scripts, the author's README/Roadmap and assets were retained from the Mac project. They were not replaced with an incomplete deployment-only directory. The `collect/` directory is retained Mac code; the separate `demo_collection/` directory contains the NUC collection scripts. Matching relative paths use the NUC version.

`analysis/2026-10-25/` contains the earlier 70-trial analysis, figures and derived JSON snapshot. It does not include the subsequently downloaded cube-position trials or all NUC runtime logs. The analysis script uses original workspace paths and is retained as its original source; it is not yet a portable reproduction command for a fresh standalone checkout. The published snapshot and figures remain available for review.

`ROBOT_SETUP.md` and `IL_PIPELINE.md` are copied into `docs/` as existing workflow context. Other copied operational documents contain historical paths and settings. Their presence does not establish that every command works from this new repository location. No robot operation, model training or dependency installation was performed during repository preparation.

## External dependencies and limits of reproduction

This is project code, not a full LeRobot fork. It imports an installed LeRobot environment along with PyTorch and other existing dependencies. The original Mac checkout is based at LeRobot commit `c364df18c54b670b6450073039aa2cd6024a6855`; the NUC's installed LeRobot commit/environment was not captured by this code download. That Mac commit identifier alone does not reproduce the NUC environment.

`support/configuration_act.py` is the downloaded ACT configuration extension, including the action-label offset. It is a reference copy, not automatically installed into LeRobot. The Mac code also includes project-related LeRobot changes outside this directory; these have not been copied into the standalone project. Exact dependency pinning and a complete portable setup guide remain reproduction work, rather than a verified feature of this first repository snapshot.

Datasets, checkpoint weights, raw rollout logs, transfer archives, caches and backup scripts remain local and are excluded. Original data and checkpoint files were not moved, overwritten or deleted. The author's two videos and diagram assets are included as showcase material. Raw data/model distribution can be decided separately.

## Local integrity check

From the standalone repository root, using Python 3:

```bash
python3 tools/verify_source_snapshot.py
```

This hashes the 25 included NUC-authoritative files and compares them with `source_provenance.json`. It does not import robot libraries, connect to hardware, or write files. After intentionally changing deployment code, update provenance deliberately; otherwise a mismatch should be investigated.

GitHub publication is pending the user's review of this local repository.
