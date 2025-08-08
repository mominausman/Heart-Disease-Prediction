# Heart-Disease-Prediction

## What this cleaned package contains
- Project source files (excluded local virtual environments and large binaries).
- `models/` — any `.pkl` model files that were present in the original archive (if any). Files included: heartattack.pkl, heart_attack_rf_model.pkl, scaler.pkl
- `inspection_report.md` — automated inspection report produced earlier.
- `requirements.txt` — inferred dependency list (best-effort).
- `smoke_test.py` — a small script to verify model loading and environment.
- A cleaned file tree without `venv`, `__pycache__`, and heavy binary site-packages.

## How to set up (example)
1. Create & activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate    # Windows (PowerShell)
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place any required model files (if you did not include them) into the `models/` folder.
4. Run the smoke test:
```bash
python smoke_test.py
```


