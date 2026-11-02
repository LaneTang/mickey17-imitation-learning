# Automatic full-task completion estimates

This analysis is separate from the November 1 **placement-time** results.
Times are observed-motion return candidates, not operator labels or verified
task successes. They include the initial inference hold and mark the onset of
a sustained return of all six channels to within 7 units of the starting pose,
after a departure and opening–closing–reopening sequence.

The figure shows all individual normal-ending candidates and their medians.
Counts are timed candidates / all trials in each original alpha group.
Undetected times are not replaced by the 40-second timeout. Abnormal endings
are excluded from timing summaries. The alpha groups contain 50 distinct
trials; supplementary attempts with the same alpha value are not pooled in.

| α | Timed / all | Median (s) | Range (s) | Abnormal endings |
| --- | --- | --- | --- | --- |
| 0.1 | 8/10 | 26.18 | 23.31–37.28 | 0 |
| 0.2 | 7/10 | 19.03 | 14.36–27.68 | 0 |
| 0.5 | 7/10 | 20.35 | 16.63–28.61 | 0 |
| 0.8 | 6/10 | 17.66 | 15.04–23.42 | 3 |
| 1.0 | 5/10 | 25.78 | 18.75–34.74 | 3 |

Alpha 0.8 has the shortest candidate median, but also three abnormal endings.
These conditional timings do not establish which alpha is best. The detector
may identify empty-hand motion or miss real completions; its thresholds have
not been validated against video. See [method and thresholds](../../docs/AUTOMATIC_COMPLETION_TIME.md).

`alpha_completion_data.json` preserves the 50 label-independent source records,
detector settings and the source report hash. Recreate PNG, editable SVG,
summary CSV and Markdown table from this compact snapshot:

```bash
python3 tools/plot_completion_times.py \
  --input analysis/2026-11-02-completion/alpha_completion_data.json \
  --output-dir /tmp/mickey17-completion-figure
```

Run from the repository root using the analysis environment in
`requirements-analysis.txt`. The first detector run used the checksum-verified
November 1 archive. The new script and this snapshot do not modify or replace
the existing placement labels, tables, or figures.
