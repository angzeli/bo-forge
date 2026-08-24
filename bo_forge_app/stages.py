"""Compatibility alias for :mod:`bo_forge_api.stages`."""

import sys

from bo_forge_api import stages as _implementation

sys.modules[__name__] = _implementation
