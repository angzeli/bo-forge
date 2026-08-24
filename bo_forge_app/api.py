"""Compatibility alias for :mod:`bo_forge_api.api`."""

import sys

from bo_forge_api import api as _implementation

sys.modules[__name__] = _implementation
