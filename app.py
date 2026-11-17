"""Streamlit deployment entry point."""
import runpy
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "dashboard" / "app" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
