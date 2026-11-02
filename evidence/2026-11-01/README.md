# Public evidence snapshot

`outcomes.json` is a compact numerical import of the checksum-verified
November 1 archive. It records 107 unique IDs, their 127 memberships, labels,
statuses, per-chunk timing and source-file SHA-256 hashes. Original images and
full raw logs are excluded. Two derived cycle trajectories are included at
six decimal places for reproducible illustrations. They are not new trials.

Source: `mickey17-outcomes-20261101.tar`

SHA-256: `f9bec52554f3706d93810b7e9e6fade4a4783c83b54151df8f5f4fb58c3ad719`

The archive hash was checked before extraction. File hashes in the registry
refer to original bytes. Shared baseline copies were compared across all files.
The raw importer verifies status and final cycle counts for primary trials.
Public analysis can reproduce calculations, but cannot independently re-watch
the unshared raw archive. Manual outcome labels are the measurements used.

`config.json` and `train_config.json` are verbatim copies from the selected
018000 checkpoint. Their paths are historical relative paths, not a portable
training command. `split.json` records the experiment's explicit 134/24 split.

`heldout_review.json` preserves the existing review's numerical results and
limitations, replacing its absolute workspace/BC checkpoint paths with
descriptive references. `heldout_episode_151.png` is an unchanged copy of
`reviews/patch4_158_act_zero_offset_official/episode_151_joints.png`. The review
selected checkpoint 018000. That checkpoint's `model.safetensors` was hashed
locally and matches all outcome manifests:
`c2de68211a91d5fda1dbd47cb57aaef8fa5945df824363ed9f93fb06147a6528`.
Thus it is relevant recorded-observation validation, not closed-loop evidence.
The 12-frame-offset review uses a different checkpoint and is not substituted.

See [results](../../docs/RESULTS.md) and [reproduction](../../docs/REPRODUCE.md).
Raw data and model checkpoints remain unchanged and local. Snapshot files and
figure-generation code are local showcase extensions, separate from the NUC
source snapshot.
