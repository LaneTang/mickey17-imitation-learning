# My Imitation Learning Practice

## 1. TL;DR

Teaching a low-cost robot to pick and place from demonstrations by **ACT** with only one local camera, then making the learned policy work under real inference latency.

## 2. Hero Demo

recollect a beautiful rollout from a 45-degree downward angle. Make a comparison in different alpha, same angle, same timestamps, 4 rollouts stitched together horizontally.

## 3. one paragraph Intro - what I did?

The task is a simple 'pick and place' task. The platform is low cost, it's SO-101 + Beelink NUC(integrated GPU)

I esdablished a e2e pipeline of Imitation Learning, from teleoperation, data collection, policy training, deployment to review.

I collected a dataset about 200 demostrations by myself. The eventual batch contains 158 demos of 16s.

I used only 1 local camera that attached on the arm wrist, when normal ACT deployment used global camera(s).

On deployment, our lerobot hardware exist real-time inference latency(~400ms), causing oscillation. I proposed an blend filter on controller to solve it, and it worked!

## 4. result

how to define a good rollout? And how I analysed it?

### trial performance via different experiment settings

[each experiment settings test 10 times to form ]

consistency / success rate / actual motor cmd curves.

1. in different **alpha** (0.1 0.2 0.5 0.7 1)

2. in different **base position**(origin, straight left 2cm, straight right 2cm)

3. in different **cube pos**(deg:0, 10, 45, 60, 80)

### offline validation figure

1. held-out demo cmd vs. ACT proposed cmd of 6 motors

### demo review figure

1. cmd trajectory of 6 motors

## 5. About this System

### Real time system diagram

1. of data flow
2. of blending mechanism and the blend filter

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
