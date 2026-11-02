# experiment_statement

This proj is just a personal project, so I won't use a really strict repeat experiment setting to measure repeat experiment. Won't be really precise, just for fun!

## basic testbench parameter

1. define each rollout set contain 10 rollouts(well, some only have 5 that's just because of my laziness, :)
2. define standard rollout seconds = 40s

## what is a good rollout

For a standard 40s-rollout process, define 4 phases:

A rollout consists of four sequential task phases:

1. **Reaching** - Move the gripper toward the cube.
2. **Picking** - Grasp and lift the cube.
3. **Transporting** - Move the cube above the destination circle.
4. **Placing** - Release the cube at the destination.

And according to my obseration, for a happy rollout, it took about 25-30s to complete 4 phases.

So we(actually only me) define rollout quality to be:

| Rank | Completed Phases | Description |
| ------ | ------------------ | ------------- |
| 0 | 0/4 | No phase completed |
| 1 | 1/4 | Reaching completed |
| 2 | 2/4 | Picking completed |
| 3 | 3/4 | Transporting completed |
| 4 | 3/4 | Going back to default position |
| 5 | 4/4 | All phases completed |

A rollout is evaluated based on the highest sequentially
completed phase within the 40-second evaluation window.

A rollout that executes normally but fails to complete any phase
receives `0/4`, not `null`.

## quality_label

I evaluate two outcomes independently:

1. **Task phase completion** (`quality_rank`)
2. **Circle placement status** (`cube_in_the_circle.rating`)

### Abnormal rollout termination

`quality_score = null` is reserved for abnormal execution
termination, such as:

- Controller tolerance violation
- Policy runtime error
- Other execution errors that prevent the rollout from continuing

These trials are treated as execution failures, not excluded from the experiment.

### Placement rating

`cube_in_the_circle.rating = null` means that the placement stage was not completed, so placement quality cannot be evaluated.
