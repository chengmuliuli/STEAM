import os

os.environ.setdefault("HF_HOME", "/data_cfs/data1/qiuguolin/hf_cache")
os.environ.setdefault(
    "HF_DATASETS_CACHE", "/data_cfs/data1/qiuguolin/hf_datasets_cache"
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE", "/data_cfs/data1/qiuguolin/hf_cache/transformers"
)

from steam_runtime_compat import *  # noqa: F401,F403,E402
