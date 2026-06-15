"""Pytest configuration for the autotrader test suite.

Ensures the project root is importable so tests can `from prepare import ...`
when run from the tests/ subdirectory.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
