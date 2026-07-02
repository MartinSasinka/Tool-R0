import os
import sys

# Make the package modules importable when running `pytest` from anywhere.
_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FOLDER not in sys.path:
    sys.path.insert(0, _FOLDER)
