"""Adds sibling project roots to sys.path so prostea can import
protoprostate.* (and, if ever needed, prostNfound's own package) directly by
relative filesystem position, without installing them. Import this module
before importing anything from those packages; importing it again is a no-op.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLINGS_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../emese/

for _name in ("prototypeProstate", "prostNfound"):
    _path = os.path.join(_SIBLINGS_ROOT, _name)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
