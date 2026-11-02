# Mickey17 · repeat-experiment definition

This is my exploratory pick-and-place study, collected October 30–31 and transferred in the verified November 1 archive. Most configurations have ten trials; base shifts and the 45°, 60° and 80° rotations have five primary trials. The standard evaluation window is 40 seconds.

## Task and ranks

The robot reaches, picks, transports and places a cube at the destination circle. Returning toward the default pose is a separate final outcome.

| `quality_rank` | Highest sequential progress |
| --- | --- |
| 0 | No phase completed |
| 1 | Reaching completed |
| 2 | Picking completed |
| 3 | Transporting completed |
| 4 | **Placed, but not yet returned** |
| 5 | **Placed and returned** |

Placement success means rank ≥ 4; placement plus return means rank 5. Neither means the cube lies entirely inside the circle. `cube_in_the_circle.rating` is an independent operator estimate of placement quality.

An abnormal termination is an execution failure and stays in the denominator, even when its rank is null. A normal rank 0 is a valid unsuccessful outcome. A missing label or an unfinished `started` manifest is unknown, not a fabricated zero. The primary study has no missing label files. Manual observations and Space-key timing include operator judgment and reaction time.

`completion_time_s` marks **placement**, not return. Report recorded individual measurements and their median with the timed sample count; omit absent times. Three primary placements have no recorded time. Do not replace them with 40 s.

## Groups and accounting

- Filtering: α = 0.1, 0.2, 0.5, 0.8, 1.0; ten trials each. α = 1 is unfiltered.
- Base position: origin (ten shared baseline trials), 2 cm left (five), 2 cm right (five primary trials at α = 0.5).
- Cube orientation: 0° (ten shared baseline trials), 10° and 30° (ten each), 45°, 60° and 80° (five each), all at α = 0.5.

The α = 0.5, origin and 0° folders contain the **same ten trials**. Their files match by SHA-256. There are 95 distinct primary trials, nine extra base-right attempts at other alphas, and three preliminary tests: 107 unique IDs across 127 folder memberships. Supplementary attempts are reported separately.

Report observed proportions with Wilson 95% intervals, actual group sizes, and controller errors independently. These small, sequentially collected groups do not establish a universally best alpha or isolate a latency cause. The older 70-trial study uses a different scoring rubric and remains separate.

This wording reconciles my clarified definitions with the recorded data. My [original statement](docs/archive/experiment-statement-author-draft-2026-11-02.md) is preserved verbatim. The historical rollout prompt has a different 1–5 legend; the current analysis uses the explicit 0–5 definition above, not that prompt. See the [report](docs/RESULTS.md) and [registry](analysis/2026-11-01/trial_registry.csv).
