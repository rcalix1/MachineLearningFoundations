## Ideas related to SVM

* Cover with XOR and NNs since same data: https://github.com/rcalix1/MachineLearningFoundations/tree/main/NeuralNets/XOR



## FIX COLAB NOTEBOOK FOR GITHUB

```

# =========================
# FIX COLAB NOTEBOOK FOR GITHUB
# =========================
import json

# change this to your notebook filename
fname = "XOR_2026.ipynb"

with open(fname, "r", encoding="utf-8") as f:
    nb = json.load(f)

# remove problematic widget metadata
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]

# also clean per-cell metadata (just in case)
for cell in nb.get("cells", []):
    if "metadata" in cell and "widgets" in cell["metadata"]:
        del cell["metadata"]["widgets"]

with open(fname, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook cleaned. Reload it and push to GitHub.")

```
