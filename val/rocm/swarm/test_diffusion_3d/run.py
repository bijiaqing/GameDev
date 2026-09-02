#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"test_common"))
from run_model import run

# select the model from the directory name and delegate compilation, execution, archiving, and validation to shared code

run(Path(__file__).resolve().parent.name)
