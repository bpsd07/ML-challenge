# ML Challenge 2026 - Business Entity Resolution

Solution for the Business Entity Resolution Challenge (Multi-Source Cross-Entity Matching).

## Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources with noisy and inconsistent fields (business names, addresses, country codes). This project implements an end-to-end Machine Learning pipeline to match records across sources (Source 2 and Source 3) back to reference records (Source 1).

### Key Features
- **Data Profiling & Auditing:** Comprehensive exploratory data analysis across training and test sources.
- **Normalization:** Robust text cleaning, legal entity abbreviation standardization, and address normalization.
- **Blocking & Candidate Generation:** Scalable candidate pair filtering using blocking keys to reduce quadratic comparison overhead.
- **Feature Engineering:** String similarity metrics, token overlap, edit distances, n-gram jaccard similarities, and address matching features.
- **Classification & Scoring:** Trained gradient boosting / ensemble models to classify match candidates and optimize F1 score.

## Directory Structure

```text
├── src/
│   ├── blocking.py              # Candidate pair blocking mechanisms
│   ├── data_audit.py            # Data quality and schema audit utilities
│   ├── features.py              # Feature extraction pipeline
│   ├── metrics.py               # Evaluation metrics (Precision, Recall, F1)
│   ├── normalization.py         # Name & address normalization logic
│   ├── test_normalization.py    # Unit tests for text normalization
│   ├── train.py                 # Model training and validation pipeline
│   └── profiling/
│       └── profile_dataset.py   # Dataset profiling script
├── reports/                     # Profiling, evaluation, and audit reports
├── requirements.txt             # Python dependencies
└── problem_statement.pdf        # Challenge problem description
```

## Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run normalization tests:
   ```bash
   python -m unittest src/test_normalization.py
   ```
