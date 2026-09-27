import json
from pathlib import Path

def build_self_contained_notebook():
    src_dir = Path("src")
    normalization_code = (src_dir / "normalization.py").read_text(encoding="utf-8")
    blocking_code = (src_dir / "blocking.py").read_text(encoding="utf-8")
    eval_code = (src_dir / "evaluate_stage2c_blocking.py").read_text(encoding="utf-8")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon Business Entity Resolution: Stage 2C Multi-Script & Address Recall Recovery\n",
                "### Objective: Recover Dravidian/Bengali/Gujarati Scripts and Address Digit/Format Misses (Target >=97% to >=98%) on Frozen Pilot B\n",
                "- Audit actual remaining scripts (Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Odia, Devanagari) across missed true links.\n",
                "- Evaluate V9 Baseline (39 strategies) on Pilot A (50k S1) and frozen independent Pilot B (50k S1).\n",
                "- Evaluate each of the 8 Stage 2C Targeted Blockers (SB01 to SB08) individually.\n",
                "- Evaluate progressive combinations and final optimized blocker union.\n",
                "- Validate on frozen independent Pilot B (seed=1337) against 10.3M candidate population.\n",
                "- Export reports/blocking_multiscript_experiments.csv, reports/blocking_multiscript_miss_analysis.md, reports/blocking_stage2c_optimization_report.md, and checkpoints/blocking/blocking_checkpoint_stage2c.json."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import shutil\n",
                "import psutil\n",
                "from pathlib import Path\n",
                "\n",
                "print('=== ENVIRONMENT & RESOURCE CHECK ===')\n",
                "ram = psutil.virtual_memory()\n",
                "print(f'Total RAM: {ram.total / (1024**3):.2f} GB | Available: {ram.available / (1024**3):.2f} GB')\n",
                "print(f'CPU Count: {os.cpu_count()}')\n",
                "\n",
                "# Setup working directory structure\n",
                "BASE_WORKING = Path('/kaggle/working/business_er') if Path('/kaggle/working').exists() else Path('./business_er')\n",
                "for sub in ['src', 'checkpoints/blocking', 'cache', 'reports', 'output', 'logs']:\n",
                "    (BASE_WORKING / sub).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "SRC_DIR = BASE_WORKING / 'src'\n",
                "sys.path.insert(0, str(SRC_DIR))\n",
                "print(f'Workspace initialized at: {BASE_WORKING}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Writing src/normalization.py\n",
                f"norm_code = {json.dumps(normalization_code)}\n",
                "with open(SRC_DIR / 'normalization.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(norm_code)\n",
                "print('Wrote normalization.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Writing src/blocking.py\n",
                f"block_code = {json.dumps(blocking_code)}\n",
                "with open(SRC_DIR / 'blocking.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(block_code)\n",
                "print('Wrote blocking.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Writing src/evaluate_stage2c_blocking.py\n",
                f"pipeline_code = {json.dumps(eval_code)}\n",
                "with open(SRC_DIR / 'evaluate_stage2c_blocking.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(pipeline_code)\n",
                "print('Wrote evaluate_stage2c_blocking.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== EXECUTING STAGE 2C MULTI-SCRIPT & ADDRESS BLOCKING OPTIMIZATION & PILOT B VALIDATION ===')\n",
                "import evaluate_stage2c_blocking\n",
                "\n",
                "# Run the complete Stage 2C evaluation, audit, and validation pipeline\n",
                "evaluate_stage2c_blocking.main()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== COPYING REPORTS & CHECKPOINTS TO WORKING ROOT FOR EXPORT ===')\n",
                "export_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')\n",
                "\n",
                "reports_src = BASE_WORKING / 'reports'\n",
                "checkpoints_src = BASE_WORKING / 'checkpoints'\n",
                "\n",
                "if reports_src.exists():\n",
                "    dest_rep = export_dir / 'reports'\n",
                "    dest_rep.mkdir(parents=True, exist_ok=True)\n",
                "    for f in reports_src.glob('*'):\n",
                "        if f.is_file():\n",
                "            shutil.copy(f, dest_rep / f.name)\n",
                "            print(f'Exported report: {f.name} ({f.stat().st_size / 1024:.1f} KB)')\n",
                "\n",
                "if checkpoints_src.exists():\n",
                "    dest_chk = export_dir / 'checkpoints'\n",
                "    dest_chk.mkdir(parents=True, exist_ok=True)\n",
                "    for f in checkpoints_src.rglob('*.json'):\n",
                "        rel = f.relative_to(checkpoints_src)\n",
                "        out_target = dest_chk / rel\n",
                "        out_target.parent.mkdir(parents=True, exist_ok=True)\n",
                "        shutil.copy(f, out_target)\n",
                "        print(f'Exported checkpoint: {rel}')\n",
                "\n",
                "print('\\n=== ALL ARTIFACTS READY FOR DOWNLOAD VIA KAGGLE CLI ===')\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("kaggle/main.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated self-contained {out_file} successfully ({out_file.stat().st_size / 1024:.1f} KB).")

if __name__ == "__main__":
    build_self_contained_notebook()
