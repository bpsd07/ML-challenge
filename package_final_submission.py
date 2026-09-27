"""
Package Final Submission Zip for ML Challenge 2026.
Builds the required archive structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import sys
import os
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STUDENT_RES = BASE_DIR / "student_resource"
OUTPUT_DIR = STUDENT_RES / "output"
TEAM_NAME = "bhanu_pratap_singh_deo"
ZIP_PATH = BASE_DIR / f"{TEAM_NAME}_submission.zip"


def create_submission_zip():
    print("=" * 80)
    print("PACKAGING FINAL SUBMISSION ARCHIVE")
    print("=" * 80)

    # 1. Verify output files
    matching_tsv = OUTPUT_DIR / "matching_results.tsv"
    candidate_tsv = OUTPUT_DIR / "candidate_pairs.tsv"

    if not matching_tsv.exists():
        raise FileNotFoundError(f"Missing required file: {matching_tsv}")
    if not candidate_tsv.exists():
        raise FileNotFoundError(f"Missing required file: {candidate_tsv}")

    print(f"Verified output files:")
    print(f"  matching_results.tsv ({matching_tsv.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  candidate_pairs.tsv  ({candidate_tsv.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Run validator
    validator_script = STUDENT_RES / "utils" / "validate_submission.py"
    test_dir = STUDENT_RES / "dataset" / "test"

    print("\nRunning submission validator...")
    import subprocess
    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(test_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator stderr:", res.stderr)

    if res.returncode != 0:
        print("WARNING: Validator reported issues! Please review above before final submission.")
    else:
        print("VALIDATION PASSED! Safe to submit.")

    # 3. Create zip file
    print(f"\nCompressing archive to: {ZIP_PATH} ...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output/
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # Add code/business_entity_resolution/src/
        src_dir = BASE_DIR / "src"
        for py_file in src_dir.glob("*.py"):
            zf.write(py_file, arcname=f"code/business_entity_resolution/src/{py_file.name}")

        # Add README and requirements
        main_readme = BASE_DIR / "README.md"
        if main_readme.exists():
            zf.write(main_readme, arcname="code/business_entity_resolution/README.md")

        req_file = BASE_DIR / "requirements.txt"
        if req_file.exists():
            zf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")

        # Add Documentation_template.md
        doc_file = STUDENT_RES / "Documentation_template.md"
        if doc_file.exists():
            zf.write(doc_file, arcname="Documentation_template.md")

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSubmission archive successfully generated:")
    print(f"  Path: {ZIP_PATH}")
    print(f"  Size: {zip_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    create_submission_zip()
