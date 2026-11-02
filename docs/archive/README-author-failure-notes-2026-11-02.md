<div align="center">

# Mickey17

**My imitation learning practice**

[Demo](https://lanetang.github.io/mickey17-imitation-learning/media/comparison.html) · [Results](#4-results) · [System](#5-about-this-system) · [Reproduce](#7-reproduction-guide)

</div>

## 1. TL;DR

I taught a low-cost robot to pick and place from demonstrations using **ACT**, then worked on chunk scheduling and a controller-side **blend filter** to make it run on a CPU.

**158 demonstrations · One wrist camera · CPU inference**

## 2. This is it!

<a href="https://lanetang.github.io/mickey17-imitation-learning/media/comparison.html"><img src="assets/alpha_comparison.gif" width="900" alt="Three independent robot demos, left to right: alpha 0.2, 0.5 and 1.0, with elapsed video time and clip-end markers"></a>

*α = 0.2, 0.5 and 1.0, left to right. Independent demos at original speed; ended clips freeze and are marked. These are illustrative recordings, not synchronized trials.*

**[Watch the full comparison](https://lanetang.github.io/mickey17-imitation-learning/media/comparison.html)** · [Single α = 0.5 recording](https://lanetang.github.io/mickey17-imitation-learning/media/landscape.html) · [Another recording, from the front](https://lanetang.github.io/mickey17-imitation-learning/media/hero.html)

## 3. Introduction — What I did?

The task is simple: "pick up a cube and place it at a circle".

I collected about 200 demonstrations myself and cleaned them into a final dataset of 158 episodes. I trained ACT through LeRobot's official pipeline with 134 training episodes and 24 held-out episodes, selecting the 18,000-step checkpoint. The policy uses one **wrist-mounted camera and robot state**.

On the NUC, logged policy inference took a median **321 ms** across 8,300 completed chunks. The robot needs commands while the next inference is running, so I used an asynchronous worker and an overlapping action queue.

The arm shook and overshot during real rollouts, so I added chunk blending, an arm filter and controller checks based on my demonstrations. Then I tested different filter settings and scene changes. ACT and its model implementation come from [the ACT authors](https://arxiv.org/abs/2304.13705) and [LeRobot](https://github.com/huggingface/lerobot). I collected the demonstrations, trained the policy, worked on execution and ran the experiments.

## 4. Results

### Experiments

I ran **95 distinct trials**, each lasting up to 40 seconds. I tested five α settings, base shifts of ±2 cm and six cube orientations. Most groups have ten trials; I stopped at five for some scene changes. **α = 1 is the unfiltered baseline.**

![Three-panel alpha outcome, placement and controller-stop comparison](analysis/2026-11-01/alpha.png)

*Numbers are trial counts.*

| α | Placement | Placed + returned | Controller stops | Placement median (timed n) |
| --- | --- | --- | --- | --- |
| 0.1 | 8/10 | 8/10 | 0/10 | 29.44 s (8) |
| 0.2 | 7/10 | 7/10 | 0/10 | 19.99 s (7) |
| 0.5 | 8/10 | 7/10 | 0/10 | 26.25 s (7) |
| 0.8 | 6/10 | 6/10 | 3/10 | 18.97 s (6) |
| 1.0 | 6/10 | 6/10 | 3/10 | 28.20 s (6) |

**My take:** Lower alpha made the arm look smoother, and the lower-alpha groups avoided controller stops in my tests. I wouldn't pick a single winner from ten runs per setting, though. The placement times above are my Space-key markers.

### Completion time

![Individual estimated completion times and medians for five alpha settings](analysis/2026-11-02-completion/alpha_completion.png)

*I estimated these times from gripper motion and the return to the starting pose. Dots show timed runs; bars show medians. These estimates haven’t been checked against video. [How I measured them](docs/AUTOMATIC_COMPLETION_TIME.md).*

### Typical failure

1. gif of real demo that shows rollout(alpha=1) with ACT overextention and ACT proposal change too much. Style like the hero demo

### Scene changes

![Base-position and cube-orientation sensitivity](analysis/2026-11-01/scene_sensitivity.png)

*At α = 0.5, small scene changes disrupted execution. Origin and 0° reuse the same ten baseline trials.*

![Flow of the 95 unique primary trials](analysis/2026-11-01/outcome_flow.png)

*Pooled accounting across all conditions. Controller stops remain failures; their completed phases were not ranked.*

The [full report](docs/RESULTS.md) includes individual times, intervals and failure categories. Nine additional base-right attempts and three preliminary tests are reported separately. The earlier 70-trial study uses different scoring.

### Offline validation figure

![Held-out episode 151 commands compared with policy predictions](evidence/2026-11-01/heldout_episode_151.png)

*Recorded held-out episode 151, using the same checkpoint as deployment. The BC trace is a historical comparator. Recorded-observation prediction does not measure physical task success.*

### Demo review figure

![Six-joint trace of the median timed rank-5 alpha-0.5 rollout](analysis/2026-11-01/trajectory_alpha_0_5.png)

*Median timed rank-5 α = 0.5 trial. The dotted line marks placement; the curves separate queued, filtered, echoed and observed values. The [unfiltered controller-stop example](analysis/2026-11-01/trajectory_alpha_1_0.png) is selected by earliest trial ID.*

## 5. About this System

### Real-time system diagram

![ACT image and state processing, asynchronous inference and controller execution](analysis/2026-11-02-readme/runtime_architecture.png)

*The worker predicts a 30-step chunk while the controller keeps executing queued commands. Only the arm and gripper move; base commands stay at zero.*

![Chunk scheduling at source cycle 24 and ready cycle 34, with a separate arm filter](analysis/2026-11-02-readme/scheduling_filter.png)

*A scheduling example: the old queue covers inference time. When the new chunk arrives, its past predictions expire and overlapping future predictions blend.*

I request another chunk about every 12 control frames. Queue blending combines predictions for the **same future cycle**. The α filter smooths commands **over time** across the first five arm channels. The gripper bypasses that filter. Raw checks run before filtering, followed by bounded-target checks.

### Dataset

![Dataset collection groups and their training and validation split](analysis/2026-11-02-readme/dataset_flow.png)

*110 demos with scene variation + 48 with motion variation = 158 demos. The connecting bands show the saved training split.*

In the first 110 demos, I changed the surroundings in the camera view while keeping the motion on the happy path. In the extra 48, I kept the scene stable and deliberately overshot during picking and placing, then recovered.

![All 158 demonstrations overlaid across the six arm and gripper command channels](analysis/2026-11-02-readme/dataset_joint_commands.png)

*Each line is one demonstration, sampled at 10 Hz for plotting. Blue is scene variation; peach is motion variation. These are recorded commands, not measured joint motion.*

This was my first time collecting demos for robot learning. Looking back, I'd include more **motion variation and recovery examples**. I saw some recovery in the hero demo, but it's still weak.

### Pipeline flow

#### Practice pipeline

![Hands-on practice from robot assembly to supervised trials and review](analysis/2026-11-02-readme/practice_pipeline.png)

*The whole process I worked through. Failed trials sent me back to the scene, demonstrations or policy.*

#### Experiment pipeline

![Demonstration collection, training, checkpoint selection and deployment](analysis/2026-11-01/pipeline.png)

*From moving the robot myself to training, held-out review and physical evaluation.*

#### Policy deployment

![Mac checkpoint review followed by NUC preflight, shadow testing and supervised rollout](analysis/2026-11-02-readme/policy_deployment.png)

*I checked the checkpoint offline first, then worked through live checks before letting the policy control the arm.*

### Platform, environment and task definition

<img src="assets/90dd8d0f4825327d1f420b8074e4b9b9.jpg" width="900" alt="SO-101 leader and follower, LeKiwi base and tabletop task">

*SO-101 leader–follower setup, wrist camera and LeKiwi base. Deployment uses the NUC CPU and zero base-velocity commands.*

The task progresses through reaching, picking, transporting and placing, followed by return. Rank 4 means placed but not returned; rank 5 includes return. Circle-overlap quality is separate. See my [experiment definition](experiment_statement.md).

## 6. Experiment narrative

Offline predictions looked promising, but physical rollouts exposed problems that recorded observations cannot show. I saw oscillation, overshoot and sensitivity to the scene. The wrist view can also hide the cube and destination during placement.

This was my first demonstration-collection effort, and the data quality and coverage could be better. Labels and placement times are my own observations.

I suspect inference delay is part of the problem: by the time a chunk is ready, the arm has already moved. I haven't isolated that cause in these tests. Looking at predictions, sent commands and measured motion together taught me more than validation loss alone.

## 7. Reproduction guide

Regenerate the figures and tables from the public numerical snapshot with Python 3.11+:

```bash
python3 -m pip install -r requirements-analysis.txt
python3 tools/analyze_outcomes.py --output-dir /tmp/mickey17-analysis
```

The [reproduction guide](docs/REPRODUCE.md) covers the training configuration, offline analysis and hardware dependencies. The evaluated deployment entry point is [rollout_act_overlap.py](deployment/rollout_act_overlap.py).

The new [dataset and system figures](analysis/2026-11-02-readme/README.md) and [completion-time chart](analysis/2026-11-02-completion/README.md) have separate reproduction commands.

## 8. Limitations and future expansion

This is a side project with small trial groups, my own labels and limited scene variation. The weights, demonstration videos and full NUC environment aren't included here.

Possible extensions include more varied demonstrations, image and demonstration augmentation, [Real-Time Chunking](https://www.pi.website/research/real_time_chunking), and VLA policies. Those are ideas for later, not work I'm starting now.

## 9. References

[ACT paper](https://arxiv.org/abs/2304.13705) · [LeRobot ACT documentation](https://huggingface.co/docs/lerobot/act) · [Attribution](docs/THIRD_PARTY_NOTICES.md) · [Source provenance](docs/SOURCE_OF_TRUTH.md)

I used Codex to help with code, analysis and presentation. The robot work, observations and editorial choices are mine.
