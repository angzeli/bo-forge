"""Accidental-write guard for trusted notebooks, not a security sandbox."""

import json
import os
import sys
from pathlib import Path


def install():
    protected = tuple(Path(p) for p in json.loads(Path(os.environ["NB_PROTECTED"]).read_text()))

    def check(path, *, ancestors=True):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if any(resolved == item or (ancestors and resolved in item.parents) for item in protected):
            raise PermissionError(f"Packaged input is read-only: {resolved}")

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if ((mode and any(flag in mode for flag in "wax+"))
                    or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                check(path, ancestors=False)
        elif event in {"os.remove", "os.rmdir", "os.chmod", "os.truncate"}:
            check(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check(args[0])
            check(args[1])

    sys.addaudithook(audit)
