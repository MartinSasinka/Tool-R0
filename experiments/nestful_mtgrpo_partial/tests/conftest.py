"""Make this experiment + the sibling (modules and test helpers) importable."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                                  # nestful_mtgrpo_partial
_SIBLING = os.path.join(os.path.dirname(_ROOT), "nestful_mtgrpo_minimal")

for p in (_ROOT, _SIBLING, os.path.join(_SIBLING, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
