"""Training and evaluation code for ReviewLens. Notebooks are thin runners over these functions.

The backend's pure-Python core (BIO tagging, text windows, CVAE definition, fidelity) is
shared, so training and serving cannot drift apart.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SEED = 42
