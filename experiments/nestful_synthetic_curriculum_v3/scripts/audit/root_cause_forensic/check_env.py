import sys
print(sys.version)
for m in ("numpy", "safetensors", "torch", "yaml", "pytest"):
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "OK"))
    except Exception as e:
        print(m, "MISSING:", e)
