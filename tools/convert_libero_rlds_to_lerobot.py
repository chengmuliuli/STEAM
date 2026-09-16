"""Convert the OpenVLA modified LIBERO RLDS data to LeRobot format.

This follows Physical Intelligence's OpenPI LIBERO converter, with the
LeRobot 0.3.x import path and add_frame(frame, task) API.
"""

from pathlib import Path
import shutil

import tensorflow_datasets as tfds
import tyro

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_NAME = "qiuguolin/libero_rlds_v3_fixed"
RAW_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)


def main(data_dir: str) -> None:
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing output dataset: {output_path}"
        )

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                task = step["language_instruction"].decode("utf-8")
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    },
                    task,
                )
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)
