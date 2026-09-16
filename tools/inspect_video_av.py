from pathlib import Path
import av

ROOT = Path("/data_cfs/data1/datasets/LIBERO_no_noops_v30_512/libero_10_no_noops_lerobot/videos")
for path in sorted(ROOT.glob("**/*.mp4")):
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = sum(1 for _ in container.decode(stream))
    print(path.name, "frames", frames, "duration", stream.duration, "time_base", stream.time_base, "size", path.stat().st_size)
