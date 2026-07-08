import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def root(*args):
    return PROJECT_ROOT.joinpath(*args)