import json
from pathlib import Path

def build_self_contained_notebook():
    src_dir = Path("src")
    normalization_code = (src_dir / "normalization.py").read_text(encoding="utf-8")
    blocking_code = (src_dir / "blocking.py").read_text(encoding="utf-8")
    eval_code = (src_dir / "evaluate_blocking_pipeline.py").read_text(encoding="utf-8")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon Business Entity Resolution: High-Recall Blocking Optimization\n",
                "### Objective: Maximize Blocking Recall (Target >=99%) on 50k Pilot Benchmark\n",
                "- Audit 13.31% missed matches from baseline across 18 specific failure modes.\n",
                "- Evaluate individual blocker contributions and unique value.\n",
                "- Progressively evaluate Versions 1 through 8 (Char N-Grams, Rare Token Pairs, Address Signatures, Phonetic Retrieval, Uncapped Indexes).\n",
                "- Validate frozen blocker on independent 50k Pilot B.\n",
                "- Save checkpoints and analytical reports."
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
                "# Writing src/evaluate_blocking_pipeline.py\n",
                f"pipeline_code = {json.dumps(eval_code)}\n",
                "with open(SRC_DIR / 'evaluate_blocking_pipeline.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(pipeline_code)\n",
                "print('Wrote evaluate_blocking_pipeline.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== EXECUTING HIGH-RECALL BLOCKING AUDIT & OPTIMIZATION ===')\n",
                "import evaluate_blocking_pipeline\n",
                "\n",
                "# Run the complete blocking evaluation and audit pipeline\n",
                "evaluate_blocking_pipeline.main()\n"
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
