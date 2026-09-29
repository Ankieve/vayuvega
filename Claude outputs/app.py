"""Entry point for a Hugging Face *Gradio* Space (free CPU).

The Space runs this file. It does not use Gradio itself: it just starts the
project's own web server (backend/server.py) on port 7860, which is the port
Hugging Face forwards to the public page.
"""
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = "7860"

runpy.run_path(str(ROOT / "backend" / "server.py"), run_name="__main__")
