# Historical experiments

These files preserve earlier approaches, not the final deployment workflow.
They were relocated without changing their bytes. Relative imports, dataset
paths and commands retain their original layout assumptions; do not run them
from their archived locations without a separate restoration and review.

| Directory | Contents |
| --- | --- |
| `deployment/` | Earlier ACT rollout, unbounded ACT, and BC rollout variants |
| `training/bc/` | Direct, residual and future-residual BC training |
| `training/act_arm_only/` | Earlier six-channel ACT training and latency-12 wrapper |
| `collection/` | Alternative Mac collection scripts; the NUC versions are in `../collection/` |
| `migration/` | One-time NUC layout migration script |

The final runtime remains [rollout_act_overlap.py](../deployment/rollout_act_overlap.py).
Its existing imports still require `deployment/preflight.py`, `shadow_act.py`,
`shadow_bc.py`, and `train/bc_model.py` and `bc_targets.py`. Those files have
not been archived or refactored. ACT diagnostics remain in `review/` and
`deployment/profile_act_*.py`.

The showcased checkpoint came from the official nine-channel LeRobot training
run, not the archived arm-only wrapper. See [training provenance](../docs/REPRODUCE.md#training-provenance).

Original NUC hashes remain in `source_provenance.json`. The separate
`source_relocations.json` maps old paths to new paths and records every moved
file's hash. Run `python3 tools/verify_source_snapshot.py` from the repository
root to check both source provenance and byte-preserving relocation.
