#!/usr/bin/env python3
"""CLI wrapper for the migrated Domino acceptance-length evaluator."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference.check_acceptance import main

if __name__ == "__main__":
    raise SystemExit(main())
