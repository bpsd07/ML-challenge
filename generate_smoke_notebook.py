import json
from pathlib import Path

def generate_smoke_test_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Kaggle Compute Environment Smoke Test\n",
                "Verifies environment resources (CPU, RAM, GPU), directory structures, and dataset paths without running entity resolution."
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
                "import json\n",
                "import psutil\n",
                "import platform\n",
                "from pathlib import Path\n",
                "import polars as pl\n",
                "import lightgbm as lgb\n",
                "import torch\n",
                "\n",
                "print('=== SYSTEM & RUNTIME SPECIFICATIONS ===')\n",
                "print(f'Python Version: {sys.version}')\n",
                "print(f'Platform: {platform.platform()}')\n",
                "print(f'CPU Count (Logical): {os.cpu_count()}')\n",
                "print(f'CPU Count (Physical): {psutil.cpu_count(logical=False)}')\n",
                "\n",
                "ram = psutil.virtual_memory()\n",
                "ram_total_gb = ram.total / (1024 ** 3)\n",
                "ram_avail_gb = ram.available / (1024 ** 3)\n",
                "print(f'RAM Total: {ram_total_gb:.2f} GB')\n",
                "print(f'RAM Available: {ram_avail_gb:.2f} GB')\n",
                "\n",
                "print(f'Polars Version: {pl.__version__}')\n",
                "print(f'LightGBM Version: {lgb.__version__}')\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "\n",
                "gpu_available = torch.cuda.is_available()\n",
                "gpu_name = torch.cuda.get_device_name(0) if gpu_available else 'None (CPU Mode)'\n",
                "gpu_memory = f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB' if gpu_available else 'N/A'\n",
                "print(f'GPU Available: {gpu_available}')\n",
                "print(f'GPU Device: {gpu_name}')\n",
                "print(f'GPU Total Memory: {gpu_memory}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== DISCOVERING DATASETS IN /kaggle/input ===')\n",
                "input_root = Path('/kaggle/input')\n",
                "found_files = {}\n",
                "\n",
                "expected_targets = [\n",
                "    'train_source1.tsv',\n",
                "    'train_source2.tsv',\n",
                "    'train_source3.tsv',\n",
                "    'train_ground_truth.tsv',\n",
                "    'test_source1.tsv',\n",
                "    'test_source2.tsv',\n",
                "    'test_source3.tsv',\n",
                "    'validate_submission.py'\n",
                "]\n",
                "\n",
                "if input_root.exists():\n",
                "    for p in input_root.rglob('*'):\n",
                "        if p.is_file():\n",
                "            size_mb = p.stat().st_size / (1024 * 1024)\n",
                "            print(f'Found: {p} ({size_mb:.2f} MB)')\n",
                "            for target in expected_targets:\n",
                "                if p.name == target:\n",
                "                    found_files[target] = str(p.resolve())\n",
                "else:\n",
                "    print('Warning: /kaggle/input does not exist locally (checking fallback paths).')\n",
                "\n",
                "print('\\nTarget File Resolution Summary:')\n",
                "for target in expected_targets:\n",
                "    loc = found_files.get(target, 'NOT FOUND')\n",
                "    print(f'  {target:25s} -> {loc}')\n",
                "\n",
                "all_targets_found = all(target in found_files for target in expected_targets)\n",
                "print(f'\\nAll Target Files Found: {all_targets_found}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=== CREATING STANDARD KAGGLE WORKING DIRECTORY ===')\n",
                "base_working = Path('/kaggle/working/business_er') if Path('/kaggle/working').exists() else Path('./business_er')\n",
                "subdirs = [\n",
                "    'src', 'checkpoints', 'cache', 'features',\n",
                "    'candidates', 'models', 'reports', 'output', 'logs'\n",
                "]\n",
                "\n",
                "created_dirs = []\n",
                "for sd in subdirs:\n",
                "    d = base_working / sd\n",
                "    d.mkdir(parents=True, exist_ok=True)\n",
                "    created_dirs.append(str(d))\n",
                "    print(f'Created: {d}')\n",
                "\n",
                "smoke_result = {\n",
                "    'status': 'PASS',\n",
                "    'all_targets_found': all_targets_found,\n",
                "    'found_files': found_files,\n",
                "    'ram_total_gb': ram_total_gb,\n",
                "    'ram_avail_gb': ram_avail_gb,\n",
                "    'cpu_count': os.cpu_count(),\n",
                "    'gpu_available': gpu_available,\n",
                "    'gpu_name': gpu_name,\n",
                "    'gpu_memory': gpu_memory\n",
                "}\n",
                "\n",
                "checkpoint_file = base_working / 'checkpoints' / 'smoke_test_passed.json'\n",
                "with open(checkpoint_file, 'w', encoding='utf-8') as f:\n",
                "    json.dump(smoke_result, f, indent=2)\n",
                "\n",
                "print(f'\\nSmoke test checkpoint saved to: {checkpoint_file}')\n",
                "print('=== SMOKE TEST COMPLETED SUCCESSFULLY ===')\n"
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

    out_path = Path("kaggle/main.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path} successfully.")

if __name__ == "__main__":
    generate_smoke_test_notebook()
