import json
from pathlib import Path


SOURCE = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot")
COMPAT = Path("/data_cfs/data1/qiuguolin/datasets/steam/libero_10_v30_compat")

episodes_path = COMPAT / "meta" / "episodes.jsonl"
stats = json.loads((SOURCE / "meta" / "stats.json").read_text())
out_path = COMPAT / "meta" / "episodes_stats.jsonl"

count = 0
with episodes_path.open() as f, out_path.open("w") as out:
    for line in f:
        if not line.strip():
            continue
        episode_index = json.loads(line)["episode_index"]
        out.write(json.dumps({"episode_index": episode_index, "stats": stats}) + "\n")
        count += 1
print(f"episodes_stats: {count}")
