#!/usr/bin/env python
"""Record A2 dataset in LeRobot format.

This script records demonstrations using the A2 policy (ViLGP3D) for
grasp and pick-and-place tasks. The dataset is saved in LeRobot format
and can be uploaded to HuggingFace Hub.

Usage:
    python -m lerobot_policy_a2.scripts.record_dataset \
        --task grasp \
        --num_episodes 100 \
        --output data/a2_grasp

    python -m lerobot_policy_a2.scripts.record_dataset \
        --task pick_and_place \
        --num_episodes 100 \
        --output data/a2_pickplace \
        --upload dgrachev/a2_pickplace_100
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Record A2 dataset in LeRobot format")

    # Task settings
    parser.add_argument("--task", type=str, default="grasp",
                        choices=["grasp", "pick_and_place"],
                        help="Task type")
    parser.add_argument("--object_set", type=str, default="train",
                        choices=["train", "test"],
                        help="Object set to use")
    parser.add_argument("--num_objects", type=int, default=8,
                        help="Number of objects per episode")

    # Recording settings
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to record")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")

    # Camera settings
    parser.add_argument("--cameras", type=str, default="front,overview,gripper",
                        help="Cameras to record (comma-separated)")
    parser.add_argument("--include_depth", action="store_true", default=True,
                        help="Include depth observations")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Image height")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Image width")

    # Output settings
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for dataset")
    parser.add_argument("--upload", type=str, default=None,
                        help="HuggingFace repo to upload dataset (e.g., dgrachev/a2_grasp)")

    # Model settings (for policy-based recording)
    parser.add_argument("--model_path", type=str,
                        default="dgrachev/a2_pretrained",
                        help="Path or HF repo for pretrained model")
    parser.add_argument("--use_rope", action="store_true", default=True,
                        help="Use RoPE in policy")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Show PyBullet GUI")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def record_episode(env, policy, args):
    """Record a single episode.

    Returns:
        dict: Episode data containing observations, actions, and metadata
    """
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
    }

    obs, info = env.reset()
    done = False
    truncated = False
    step = 0

    while not (done or truncated) and step < args.max_steps:
        # Get action from policy
        # For demonstration recording, we use the pretrained policy
        # In practice, this would call policy.select_action()
        action = env.action_space.sample()  # Placeholder - use policy

        # Store observation
        episode_data["observations"].append(obs)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Store action and results
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["dones"].append(done or truncated)
        episode_data["infos"].append(info)

        obs = next_obs
        step += 1

    return episode_data


def save_lerobot_format(episodes, output_dir, args):
    """Save episodes in LeRobot format.

    LeRobot datasets are stored as:
    - data/episode_XXXXXX/ containing parquet files
    - meta/info.json with dataset metadata
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset info
    info = {
        "task": args.task,
        "object_set": args.object_set,
        "num_episodes": len(episodes),
        "cameras": args.cameras.split(","),
        "image_size": [args.image_height, args.image_width],
        "fps": 30,
    }

    # Save info
    import json
    meta_dir = output_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save episodes
    data_dir = output_path / "data"
    data_dir.mkdir(exist_ok=True)

    for i, ep_data in enumerate(episodes):
        ep_dir = data_dir / f"episode_{i:06d}"
        ep_dir.mkdir(exist_ok=True)

        # Save observations as images
        for j, obs in enumerate(ep_data["observations"]):
            # Save each camera image
            if "pixels" in obs:
                for cam_name, img in obs["pixels"].items():
                    img_path = ep_dir / f"frame_{j:06d}_{cam_name}.png"
                    from PIL import Image
                    Image.fromarray(img).save(img_path)

        # Save actions and other data as parquet
        import pandas as pd
        df = pd.DataFrame({
            "action": ep_data["actions"],
            "reward": ep_data["rewards"],
            "done": ep_data["dones"],
        })
        df.to_parquet(ep_dir / "episode_data.parquet")

    print(f"Dataset saved to {output_path}")


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    print("Creating A2 environment...")
    try:
        from lerobot.envs.a2 import A2Env
    except ImportError:
        print("Error: lerobot[a2] not installed. Please install with:")
        print("  pip install lerobot[a2]")
        sys.exit(1)

    env = A2Env(
        task=args.task,
        object_set=args.object_set,
        num_objects=args.num_objects,
        camera_name=args.cameras,
        observation_height=args.image_height,
        observation_width=args.image_width,
        include_depth=args.include_depth,
        episode_length=args.max_steps,
        gui=args.gui,
    )

    # Load policy (optional - for demonstration)
    policy = None
    if args.model_path:
        print(f"Loading policy from {args.model_path}...")
        try:
            from lerobot_policy_a2 import A2Policy
            policy = A2Policy.from_pretrained(args.model_path)
            policy.eval()
        except Exception as e:
            print(f"Warning: Could not load policy: {e}")
            print("Recording with random actions instead.")

    # Record episodes
    print(f"Recording {args.num_episodes} episodes...")
    episodes = []
    for i in tqdm(range(args.num_episodes)):
        ep_data = record_episode(env, policy, args)
        episodes.append(ep_data)

        if ep_data["infos"][-1].get("is_success", False):
            print(f"  Episode {i}: Success")
        else:
            print(f"  Episode {i}: Failed")

    # Save dataset
    print("Saving dataset...")
    save_lerobot_format(episodes, args.output, args)

    # Upload to HuggingFace
    if args.upload:
        print(f"Uploading to {args.upload}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=args.output,
                repo_id=args.upload,
                repo_type="dataset",
            )
            print(f"Dataset uploaded to https://huggingface.co/datasets/{args.upload}")
        except Exception as e:
            print(f"Error uploading: {e}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
