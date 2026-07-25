import subprocess
import sys
from pathlib import Path

for m in ("transformers", "peft", "trl"):
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "OK"))
    except Exception as e:
        print(m, "MISSING:", type(e).__name__)

V3 = Path(__file__).resolve().parents[3]
print("tests dir:")
for p in sorted((V3 / "tests").glob("*.py")):
    print(" ", p.name)
