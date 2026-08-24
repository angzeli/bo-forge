"""Compatibility alias for :mod:`bo_forge_api.cli`."""

import sys

from bo_forge_api import cli as _implementation

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
