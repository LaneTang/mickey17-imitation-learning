# Mickey17: outcomes, hybrid ACT, and showout plan

Recorded: 2026-10-25. Status: proposed experiments; no deployment implementation or robot execution authorized by this document.

## User goals

1. Analyze existing rollout outcomes and produce clear, attractive figures.
2. Design repeat experiments comparing ACT variants, especially latency-aware ACT with a 12-frame action-label offset.
3. Test the user's hybrid idea: use ordinary zero-offset ACT for the first inference/startup chunk, then use offset-12 ACT from the second inference onward.
4. Incorporate the user's handwritten observations when supplied. Those observations are pending; do not invent their contents.
5. Finish with a reliable showout demo and a polished GitHub repository explaining the work and evidence.

## Why the hybrid idea is plausible

At 30 Hz, 12 frames represent 0.4 seconds. Offset-12 ACT maps an observation at frame t to commands t+12 through t+41. During offline review, the intervening motion is supplied by the recorded demonstration. During autonomous startup, the current controller holds the initial pose while inference runs. A target learned for a moving expert 0.4 seconds later may be too far ahead of a robot that remained still. This is a hypothesis, not an established diagnosis of the earlier failure.

A zero-offset startup chunk could supply the missing initial motion. Offset-12 predictions should then use fresh observations of the moving robot and retain their future timestamps. A hybrid will not necessarily solve errors after startup: the future policy does not receive the intervening queued action sequence as an explicit input, and filtering, clipping, tracking error, and model errors can make actual motion differ from demonstration motion.

The existing queue also aligns actions by control-cycle count while nominal timestamps use seconds. Actual loop frequency and camera age need checking: twelve executed cycles need not equal exactly 0.4 wall-clock seconds. This can matter even after startup.

## Evidence and limits

- `lerobot/Atom's/Imitation Learning/README.md` documents the t+12 through t+41 training targets.
- `reviews/patch4_158_act_latency12/review.json` selects offset-12 checkpoint 004000. Its sampled first-action MAE is about 4.50 calibrated position units and ten-step chunk MAE about 4.66. These average errors are against future demonstration commands; they are not directly comparable to same-frame targets.
- That review explicitly uses recorded observations and makes no closed-loop success claim. A held-out demonstration means an episode excluded from training; it is not an autonomous robot rollout.
- `deployment/act_overlap_queue.py` preserves source+12 timing even for the first offset-12 chunk; zero-offset chunk one is instead anchored at its ready cycle.
- `deployment/rollout_act_overlap.py` checks the preview's first raw proposal against the current pose before motion. A future target could fail that preview check before any startup chunk can execute. Examine the original error log before changing this behavior; retain controller protection.
- Current zero-offset outcome manifests specify nine state/action channels; the reviewed offset-12 checkpoint is arm-only. A direct comparison is a deployment-system comparison, not an isolated offset ablation. An ablation changes one factor while keeping other factors fixed.
- Current 70-run outcomes contain only zero-offset ACT. They provide no empirical success rate for offset-12 or hybrid ACT.
- NUC inference in the current groups is roughly 0.32 seconds, versus the 0.4-second offset. Offset should be assessed against observation-to-command timing, including acquisition, scheduling, and communication, rather than inference time alone.

## Proposed hybrid behavior (not implemented)

1. Verify the exact two checkpoints, processors, joint order, calibration units, image preprocessing, state dimensions, and NUC code versions. Record both hashes.
2. Obtain a fresh observation; compute the zero-offset startup chunk while holding the initial pose, as ordinary ACT already does.
3. Start executing that chunk. Obtain a fresh observation when starting inference two, now using offset-12 ACT.
4. Keep the startup actions scheduled while the future chunk computes and until its first applicable target time. Future action k belongs at observation_time + (12+k)/30 seconds. Never re-anchor it merely because inference finished.
5. Blend only actions assigned to the same target time. Preserve filtering state across the handoff; retain existing proposal/step/lead checks and unchanged base-zero behavior. Log gripper timing as well as arm continuity.
6. If inference is late, expired actions must not be executed. If startup coverage expires before the future schedule supplies an action, stop under an explicit gap policy rather than silently jumping to a future pose.

Two models may have different input dimensions and processors, so switching is not just changing an offset flag. One global queue offset cannot represent both policies correctly without a design change. Load/warm both before motion and measure actual NUC timing; memory/CPU costs remain to be measured.

## Repeat experiment design

First resolve the handwritten observations and original offset-12 stop logs. Classify failures as preview/startup, first moving target, policy handoff, or later task behavior. Do not assign a cause from the score alone.

Then propose three conditions under one fixed lighting setup and alpha 0.5:

| Condition | Startup | Subsequent inference | Purpose |
| --- | --- | --- | --- |
| A | Zero-offset ACT | Zero-offset ACT | Reference deployment |
| B | Offset-12 ACT | Offset-12 ACT | Test the original latency-aware deployment |
| C | Zero-offset ACT | Offset-12 ACT | Test whether startup bridging helps |

Before full trials, use offline timing replay, live observation checks and shadow review, then short supervised trials according to IL_PIPELINE.md. A known failing startup must be understood before repeating it mechanically. Shadow means a human controls movement while policy predictions are logged.

After those checks, target ten attempts per condition, interleaved in small blocks rather than completing one policy first. Use a balanced sequence such as ABC, BCA, CAB repeated three times, followed by one randomly ordered ABC block. Record exact order. Hold checkpoint, mat/base/cube placement, starting pose, alpha, timeout and lighting fixed. If testing a second lighting condition later, repeat the same design within that condition.

Primary outcome: quality >=4 within 40 seconds. Secondary: quality=5 within 30 seconds. Also report controller stops, startup/handoff stop location, quality distribution and circle coverage. Quality null counts as failure under the user's current rubric; a missing label file remains unknown. Keep full completion and controller error as separate fields because a stop can occur after task completion.

For each trial log: condition, sequence/block, both checkpoint hashes where applicable, code/settings/limits hashes, lighting, scene/start-pose identifier, alpha, observation and inference timestamps, target timestamps, queue coverage/expired actions, raw/filtered/sent targets, clipping, stop reason/time, quality and circle labels. Record measured completion time separately if available; 40-second log duration is not completion time. Use a derived analysis table or sidecar annotations for retrospective corrections; preserve original rollout data.

Keep the two existing daylight alpha-0.5 sets visible separately: they assess repeatability. Compare lighting descriptively; the earlier grouped experiment did not isolate a causal lighting effect. Ten trials per setting support exploratory decisions, not precise reliability claims.

## Current outcome figure rules

Source: `Mickey17 remote/outputs/outcome!`, read-only. Recompute from each manifest and label, rather than transcribing previous chat totals.

- Denominator: all attempts, including null labels and controller errors.
- Quality null: explicit failure convention, displayed separately from a scored incomplete task.
- Entire label file missing: unknown, displayed separately.
- Circle rating: 0=100%, 1=95%, 2=90%, 3=66%, 4=33%, 5=0%; lower is better. Null remains a distinct unplaced/unrated category, not a measured 0% coverage.
- Quality score: higher is better. Do not average the two scales together.
- Error count may overlap full completion. Preserve that overlap.
- No continuous exact completion times are inferred from the quality labels.

## Showout demo and GitHub deliverables

Demo: choose the most reliable measured setting in the intended demo lighting, use a repeatable reset and visible task target, present a representative successful sequence, and prepare a recorded fallback. Show both successes and failures in the evidence. Demo reliability and a research comparison are separate goals; a hybrid remains experimental until tested.

Repository: concise project story; hardware/setup photo and diagram; task/rubric; dataset and split provenance; BC/ACT/offset/hybrid methods; reproducible analysis commands and figures; deployment architecture; checkpoint/config hashes; measured timing and outcomes; failure examples and limitations; demo video. Clearly identify upstream LeRobot versus project-specific contributions. Review licensing, privacy, data/checkpoint hosting and large-file exclusions before publishing. No repository publication is authorized yet.

## Next manageable step

Review the handwritten log together, map its observations to the startup/handoff hypotheses above, and select the first diagnostic. Implementation, additional training and robot operation require a subsequent explicit request.
