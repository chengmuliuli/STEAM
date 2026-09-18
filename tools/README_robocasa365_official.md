# Official STEAM SFT with RoboCasa365 pretrain data

This entry point keeps RLinf's official STEAM SFT implementation unchanged.
Only the RoboCasa365 LeRobot data is adapted into a separate local view.

## Prepare the data view

The source dataset is not modified. The adapter:

1. removes the incomplete `robot0_agentview_left` camera from metadata;
2. expands the task sidecar into a row-level string `task` field;
3. removes the incompatible physical `task_index` column;
4. rounds timestamps to the dataset's 20 FPS grid (`0.05` seconds).

Example:

```bash
STEAM_SOURCE=/data-cfs/data1/datasets/robocasa365-pretrain-mg \
STEAM_TARGET=/data-cfs/data1/qiuguolin/robocasa365-pretrain-mg-official-steam-task-ts \
STEAM_ROUND_TIMESTAMPS=1 \
python tools/prepare_robocasa365_official_task_view.py
```

## Start official training

```bash
bash tools/run_official_robocasa365_steam_sft.sh
```

The default configuration uses 32 bins, micro-batch size 32, global batch size
512, 16,000 optimizer steps, and saves every 1,000 steps. Override standard
RLinf/Hydra arguments after the wrapper, for example:

```bash
bash tools/run_official_robocasa365_steam_sft.sh \
  runner.max_steps=1 runner.save_interval=1
```

Set `RLINF_ROOT`, `STEAM_ROBOCASA_DATA`, `STEAM_LOG_ROOT`, `STEAM_VISION_MODEL`,
and `STEAM_LANGUAGE_MODEL` for another machine. The runtime compatibility shim
is only for the installed dependency versions; it does not change the STEAM
model, loss, sampler, or optimizer.

