"""Regenerate every manuscript figure: python figures_src/make_all.py"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRIPTS = ["fig1_coffee_ring", "fig2_setup", "fig3_data_overview", "fig4_ssim"]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name in SCRIPTS:
        if only and not any(name.startswith(o) for o in only):
            continue
        importlib.import_module(name).main()
