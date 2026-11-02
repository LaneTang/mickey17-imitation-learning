# My Imitation Learning Practice

## 1. TL;DR

Teaching a low-cost robot to pick and place from demonstrations by **ACT** with only one local camera, then introduce a **blend filter** making the learned policy work under real inference latency.

## 2. Hero Demo

recollect a beautiful rollout from a 45-degree downward angle. Make a comparison in different alpha, same angle, same timestamps, 4 rollouts stitched together horizontally.

## 3. Introduction - What I did?

The task is a simple 'pick and place' task. The platform is low cost, it's SO-101 + Beelink NUC(integrated GPU)

I esdablished a e2e pipeline of Imitation Learning, from teleoperation, data collection, policy training, deployment to review.

I collected a dataset about 200 demostrations by myself. The eventual batch contains 158 demos of 16s.

I used only 1 local camera that attached on the arm wrist, when normal ACT deployment used global camera(s).

During real-world deployment, ACT inference on my low-cost hardware takes approximately 400 ms. My hypothesis is that, while inference is running, the robot continues moving, creating a temporal mismatch between the observation used for prediction and the robot's state when the new action chunk is executed.

Without filtering (α = 1), I observed severe oscillations that frequently triggered controller tolerance violations and sometimes prevented the rollout from completing.

I implemented a controller-side blending filter to smooth the action commands. It significantly improved execution stability in my initial tests, although it does not eliminate the inference latency itself.

## 4. result

Each configuration will be tested over 10 repeated trials.

Metrics:

Task success rate

Rollout completion rate

Execution consistency

Actual motor command trajectories

### Experiments

**Filter Ablation & Sensitivity Analysis**: Different α values (0.1, 0.2, 0.5, 0.8, 1.0), where α = 1 represents the unfiltered baseline.

**Base Position Robustness**: Original position, 2 cm left, and 2 cm right.

**Cube Orientation Generalization**: 0°, 10°, 30°, 45°, 60°, and 80°.

A rollout terminated by a controller tolerance violation will be recorded as an unsuccessful trial rather than excluded from the evaluation.

### offline validation figure

1. held-out demo cmd vs. ACT proposed cmd of 6 motors

### demo review figure

1. cmd trajectory of 6 motors

## 5. About this System

### Real time system diagram

1. of data flow
2. of blending mechanism by blend filter
3. of a typical blending scheduling

### Pipeline flow

1. of the pipeline of experiment

### Platform, environment and task definition

1. img of platform

    Platform details

2. img of environment

    environment statement

task definition details.

## 6. Experiment narrative

1. result measurement by tester's label, might not be really accurate
2. didn't cover enough test of generalization.
3. first time collect demostration for IL, some demo's quality doesn't good enough.
4. **Hypothesis on execution instability:** Inference latency may introduce temporal misalignment between observations and executed action chunks, contributing to command discontinuities and oscillation. The blending filter mitigates the observed instability, but the underlying causal mechanism has not been independently verified.

## 7. Reproduction guide

I'll codex it.

## 8. Limitation and Future expansion

1. I wanna explore the theoretical solution of latency, like **Real-Time Chunking** by *Physical Intelligence*.[citation]
2. I wanna try what if I use **Data Augmentation** of image processing(for e.g. Sharpening & Brightness), to train some variant of ACT, what it's performance will be.
3. I also can try **Demonstration Augmentation**

## 9. Reference

1. ACT paper
2. Lerobot ACT doc
3. GPT 6.1-sol
4. my taste
