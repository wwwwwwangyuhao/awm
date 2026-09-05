"""Base project settings.

The package root is derived from this file location.  Launching or renaming a
checked-out package therefore never rewrites source code just to update paths.
"""
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[1])
BASE_ENV = "dssat11"

__all__ = ["BASE_DIR", "BASE_ENV"]
