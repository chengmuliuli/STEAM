from pathlib import Path
import inspect
import json
import importlib.metadata

import pyarrow.parquet as pq
from lerobot.datasets import utils
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


ROOT = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot")

print("lerobot version:", importlib.metadata.version("lerobot"))
print("metadata init:")
print(inspect.getsource(LeRobotDatasetMetadata.__init__))
print("get_episode_chunk:")
print(inspect.getsource(LeRobotDatasetMetadata.get_episode_chunk))

print("lerobot load_tasks source:")
print(inspect.getsource(utils.load_tasks))
print("lerobot load_episodes source:")
print(inspect.getsource(utils.load_episodes))
print("lerobot load_episodes_stats source:")
print(inspect.getsource(utils.load_episodes_stats))
print("lerobot __getitem__ source:")
print(inspect.getsource(LeRobotDataset.__getitem__))
print("LeRobotDatasetMetadata source:")
print("get_data_file_path:")
print(inspect.getsource(LeRobotDatasetMetadata.get_data_file_path))
print("get_video_file_path:")
print(inspect.getsource(LeRobotDatasetMetadata.get_video_file_path))
print("get_episodes_file_paths:")
print(inspect.getsource(LeRobotDataset.get_episodes_file_paths))
table = pq.read_table(ROOT / "meta" / "tasks.parquet")
print("tasks schema:", table.schema)
print("tasks rows:", table.to_pylist()[:5])
episodes = pq.read_table(ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
print("episodes schema:", episodes.schema)
print("episodes rows:", episodes.to_pylist()[:3])
print("info:", json.loads((ROOT / "meta" / "info.json").read_text()))
rows = episodes.to_pylist()
for key in [
    "data/chunk_index",
    "data/file_index",
    "videos/observation.images.image/file_index",
    "videos/observation.images.wrist_image/file_index",
]:
    print(key, sorted(set(row[key] for row in rows)))
