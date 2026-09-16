from pathlib import Path
import json

import pyarrow.parquet as pq


SRC = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot")
DST = Path("/data_cfs/data1/qiuguolin/datasets/steam/libero_10_v30_compat")


def main() -> None:
    episode_files = sorted((SRC / "meta" / "episodes").glob("**/*.parquet"))
    rows = []
    for path in episode_files:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda item: item["episode_index"])

    target = DST / "meta" / "episodes.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)
    print(f"wrote {target}")
    print(f"episodes: {len(rows)}")


if __name__ == "__main__":
    main()
