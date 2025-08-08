# smoke_test.py
# Simple checks: Python imports, model loading (if model exists) and a hello message.

import sys
import os
print("Python version:", sys.version.split()[0])

# Check installed imports
try:
    import joblib
    print("joblib import OK")
except Exception as e:
    print("joblib import failed:", e)

# Try to load a model if present in models/
models_dir = os.path.join(os.path.dirname(__file__), "models")
if os.path.isdir(models_dir):
    pkls = [f for f in os.listdir(models_dir) if f.lower().endswith('.pkl')]
    if pkls:
        print("Found model files:", pkls)
        for p in pkls:
            path = os.path.join(models_dir, p)
            try:
                obj = joblib.load(path)
                print(f"Loaded {p} (type: {{}})".format(type(obj)))
            except Exception as e:
                print(f"Failed to load {p}:", e)
    else:
        print("No .pkl model files in models/ — place required models there.")
else:
    print("models/ directory not present.")

print("Smoke test finished.")
