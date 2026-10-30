"""Preview or train separate ACT with commands starting 12 frames after observation.

Preview: python train/train_act_latency12.py --experiment experiment.json
Train:   python train/train_act_latency12.py --experiment experiment.json --train
"""

from train_act import main


if __name__ == "__main__":
    main(action_offset_frames=12)
