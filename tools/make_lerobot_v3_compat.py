from pathlib import Path
import json
import shutil

import pyarrow.parquet as pq


SRC = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot")
DST = Path("/data_cfs/data1/qiuguolin/datasets/steam/libero_10_v30_compat")


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"Refusing to overwrite {DST}")

    (DST / "meta").mkdir(parents=True)
    for name in ("data", "videos"):
        (DST / name).symlink_to(SRC / name, target_is_directory=True)
    (DST / "meta" / "episodes").symlink_to(SRC / "meta" / "episodes", target_is_directory=True)
    for name in ("info.json", "stats.json"):
        shutil.copy2(SRC / "meta" / name, DST / "meta" / name)

    rows = pq.read_table(SRC / "meta" / "tasks.parquet").to_pylist()
    with (DST / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda item: item["task_index"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"created {DST}")
    print(f"tasks: {len(rows)}")


if __name__ == "__main__":
    main()
