# Mickey17 Robot and Imitation Learning Setup

Last reviewed: 2026-10-06

## 1. Computer and Network Setup

- My main computer is a Mac. I use Codex on the Mac for project work.
- The robot computer is a NUC named `team13-MINI-S` in the terminal output.
- I connect from the Mac to the NUC remotely using Tailscale and SSH.
- The robot hardware and USB devices remain connected to the NUC.
- I want to operate the workflow from the Mac. Commands that control hardware may run on the NUC over SSH.

## 2. NUC Software Environment

- LeRobot repository on the NUC: `~/Projects/lerobot`
- Python environment shown in terminal: Conda environment `(lerobot)`
- NUC CPU observed: Intel N95, 4 CPUs
- Memory observed: about 11 GiB
- Graphics observed: Intel Alder Lake-N UHD Graphics; `torch.cuda.is_available()` returned `False`.

## 3. Robot Hardware

The robot setup has three main parts:

1. **Leader arm**
   - SO-101 arm used for teleoperation.
   - It is physically separate from the follower arm and LeKiwi base.

2. **Follower arm and base**
   - SO-101 follower arm mounted on the LeKiwi base.
   - NUC host command previously used `/dev/ttyACM0` for the robot port.

3. **Cameras**
   - Wrist camera is connected to the NUC and used for the imitation-learning task.
   - Port: `/dev/video0`.
   - Resolution: 640 × 480 30 FPS.
   - A second camera mounted on the base is not currently connected to a device and is not part of the first task.

## 4. Learning Goal

- The learning task is imitation learning using behavior cloning.
- The initial goal is to train a policy from demonstrations of the SO-101 arm.
- In the recorded dataset, observations include the wrist image and robot state; actions contain the demonstrated commands.
- The first behavior-cloning experiment focuses on the six arm joints.
- Dataset records may also contain three base-velocity fields. Confirm the training configuration before including or excluding those fields.
- Demonstration collection and policy training are separate stages. The NUC is used to operate the robot and collect data; training can be done later on a suitable machine.

## 5. Current Data Status

- Patch 2 is in `outputs/2nd_patch`.
- A cleaned copy with complete demonstrations numbered sequentially is in `outputs/2nd_patch/cleaned_renumbered`.
- It contains 11 complete demonstrations. Five zero-frame captures were excluded from that copy.
- The original demonstration folders were preserved.