"""Runtime-only compatibility for the installed LeRobot video backend.

This file is not part of RLinf or LeRobot.  It is loaded only when the
launcher prepends this directory to PYTHONPATH.  It makes torchcodec's
availability probe fail when its FFmpeg shared libraries are unavailable and
restores the old torchvision.io.VideoReader interface with PyAV.
"""

import builtins
import importlib.util
import sys


_original_find_spec = importlib.util.find_spec


def _find_spec(name, package=None):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec


def _install_pyav_video_reader():
    """Restore torchvision.io.VideoReader using the already-installed PyAV."""
    try:
        import av
        import torch
        import torchvision
    except Exception:
        return

    if hasattr(torchvision.io, "VideoReader"):
        return

    class _PyAVVideoReader:
        def __init__(self, path, stream_name="video"):
            del stream_name
            self.container = av.open(str(path))
            self.stream = self.container.streams.video[0]

        def seek(self, timestamp, keyframes_only=False):
            del keyframes_only
            time_base = float(self.stream.time_base)
            offset = int(float(timestamp) / time_base)
            self.container.seek(offset, stream=self.stream, backward=True)

        def __iter__(self):
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * self.stream.time_base)
                data = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                yield {"pts": timestamp, "data": data}

    torchvision.io.VideoReader = _PyAVVideoReader


_original_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "torchvision" or name.startswith("torchvision."):
        torchvision = sys.modules.get("torchvision")
        if (
            torchvision is not None
            and hasattr(torchvision, "io")
            and not hasattr(torchvision.io, "VideoReader")
        ):
            _install_pyav_video_reader()
    return module


# Do not import torch/torchvision in every Ray worker at interpreter startup.
builtins.__import__ = _import

