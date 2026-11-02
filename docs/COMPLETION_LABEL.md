# Manual placement timing and historical label prompt

The existing local `deployment/rollout_act_overlap.py` has a Space-key marker and format-version-3 labels. This extension was added after the downloaded NUC source snapshot. Its hash is tracked in `local_extensions.json`; the original NUC hashes remain unchanged.

During a supervised rollout, the first detected Space records `completion_time_s`. It includes initial inference hold, operator reaction time and approximately one polling-cycle uncertainty. It marks placement/release, not return. Later presses are ignored and a recorded marker survives a later failure. No Space leaves a null time. This is not automatic vision timing.

The terminal is polled without blocking and restored before rating input and cleanup. Ctrl-C remains available. The saved progress field is `quality_rank`, not legacy `quality_score`; circle rating is independent.

**Schema discrepancy:** the historical local prompt accepts 1–5 with 1=no phase and 5=placing. The completed study's author explicitly clarified a 0–5 rubric: 0=no phase, 1=reach, 2=pick, 3=transport, 4=placed but not returned, 5=placed and returned. Recorded labels include rank 0. Current analysis uses that clarified study definition. The prompt is preserved as existing runtime code; this showcase does not modify deployment behavior. Do not use the historical prompt as the study's scoring specification.

The November 1 labels contain placement times, but their existence does not establish that the current local script is byte-identical to the version installed on the NUC. Missing ratings and missing times stay unknown/absent in analysis.

Hardware-free verification (Python 3.10+):

```bash
python3 -m unittest discover -s tests -v
```

Keyboard/rating tests use pseudo-terminals and temporary files without importing policy libraries or connecting to equipment. See the [current experiment definition](../experiment_statement.md).
