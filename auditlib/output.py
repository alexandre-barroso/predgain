"""Guard generated leaves; replace files atomically without following symlinks."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .data import require


def guarded(path):
    path = Path(path)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "Output parent must be a real directory")
    require(not path.is_symlink(), f"Refusing symlink output: {path.name}")
    require(not path.exists() or path.is_file(), f"Output is not a regular file: {path.name}")
    # Hard links are unnecessary for outputs and could make later handling unsafe.
    require(not path.exists() or path.stat().st_nlink == 1, f"Refusing hard-linked output: {path.name}")
    return path


def atomic_file(path, writer):
    path = guarded(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".audit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        guarded(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    atomic_file(path, lambda stream: stream.write(payload))


def write_arrays(path, **arrays):
    atomic_file(path, lambda stream: np.savez_compressed(stream, **arrays))
