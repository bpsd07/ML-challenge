import json
from pathlib import Path

def build_self_contained_notebook():
    src_dir = Path("src")
    normalization_code = (src_dir / "normalization.py").read_text(encoding="utf-8")
    blocking_code = (src_dir / "blocking.py").read_text(encoding="utf-8")
    features_code = (src_dir / "features.py").read_text(encoding="utf-8")
    metrics_code = (src_dir / "metrics.py").read_text(encoding="utf-8")
    train_code = (src_dir / "train_stage3_matcher.py").read_text(encoding="utf-8")
    chk_2c_file = Path("checkpoints/blocking/blocking_checkpoint_stage2c.json")
    chk_2c_code = chk_2c_file.read_text(encoding="utf-8") if chk_2c_file.exists() else "{}"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon Business Entity Resolution: Stage 3 Candidate Enrichment, Hard-Negative Mining & LightGBM\n",
                "### Objective: Build the highest-precision candidate scoring and ranking layer for Macro F0.5\n",
                "- Frozen Blocker Baseline: 47 strategies, 97.27% recall on Pilot B.\n",
                "- Feature Engineering: 60 dense similarity and provenance features.\n",
                "- Hard-Negative Mining: Multi-priority negative sampling (blocker votes, name/address collisions, transliterations).\n",
                "- LightGBM Classifier: Conservative regularized GBDT binary classifier with early stopping.\n",
                "- Entity-Level Decision Policy: Empirical optimization directly maximizing Macro F0.5 per S1 (including singleton guard).\n",
                "- Export reports and checkpoints to /kaggle/working."
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
                "for sub in ['src', 'checkpoints/blocking', 'checkpoints/ml', 'cache', 'reports', 'output', 'logs']:\n",
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
                "# Writing checkpoints/blocking/blocking_checkpoint_stage2c.json\n",
                f"chk_json = {json.dumps(chk_2c_code)}\n",
                "with open(BASE_WORKING / 'checkpoints/blocking/blocking_checkpoint_stage2c.json', 'w', encoding='utf-8') as f:\n",
                "    f.write(chk_json)\n",
                "print('Wrote Stage 2C blocking checkpoint successfully.')\n"
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
                "# Writing src/features.py\n",
                f"feats_code = {json.dumps(features_code)}\n",
                "with open(SRC_DIR / 'features.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(feats_code)\n",
                "print('Wrote features.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Writing src/metrics.py\n",
                f"mets_code = {json.dumps(metrics_code)}\n",
                "with open(SRC_DIR / 'metrics.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(mets_code)\n",
                "print('Wrote metrics.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Writing src/train_stage3_matcher.py\n",
                f"pipeline_code = {json.dumps(train_code)}\n",
                "with open(SRC_DIR / 'train_stage3_matcher.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(pipeline_code)\n",
                "print('Wrote train_stage3_matcher.py successfully.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== EXECUTING STAGE 3 CANDIDATE ENRICHMENT, HARD-NEGATIVE MINING & LIGHTGBM ===')\n",
                "import train_stage3_matcher\n",
                "\n",
                "# Run the complete Stage 3 pipeline\n",
                "train_stage3_matcher.main()\n"
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
                "    for f in checkpoints_src.rglob('*'):\n",
                "        if f.is_file():\n",
                "            rel = f.relative_to(checkpoints_src)\n",
                "            out_target = dest_chk / rel\n",
                "            out_target.parent.mkdir(parents=True, exist_ok=True)\n",
                "            shutil.copy(f, out_target)\n",
                "            print(f'Exported checkpoint: {rel} ({f.stat().st_size / 1024:.1f} KB)')\n",
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
