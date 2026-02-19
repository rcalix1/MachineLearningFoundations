# Tiny SVD Example — README.md Format

This README shows **two contrasting ways to compute singular values**:

1. **Direct algorithmic method** (using eigenvalues of \(X^T X\))  
2. **NumPy's built‑in SVD** (which includes the automatic sorting step)

The contrast helps make it clear **where singular values come from** and **why SVD implementations sort them in descending order**.

---

## 📌 Matrix Used in This Example
```python
import numpy as np

X = np.array([[3.0, 1.0],
              [2.0, 2.0]])
print("Matrix X:", X)
```

---

# 1. Algorithmic SVD (Manual Computation)

This section computes singular values via the mathematical pipeline:

\[
X^T X 
ightarrow 	ext{eigenvalues} 
ightarrow 	ext{singular values}
\]

### **Step A — Compute \(X^T X\)**
```python
XtX = X.T @ X
print("X^T X:", XtX)
```

### **Step B — Compute eigenvalues of \(X^T X\)**
Eigenvalues may come out **unordered** depending on the algorithm.
```python
vals, vecs = np.linalg.eig(XtX)
print("Eigenvalues (unordered):", vals)
print("Eigenvectors:", vecs)
```

### **Step C — Singular values are the square roots of eigenvalues**
```python
singular_values_algo = np.sqrt(np.abs(vals))
print("Singular values (unordered):", singular_values_algo)
```

### **Step D — Sort singular values in descending order**
Every real SVD implementation performs this step.
```python
idx = np.argsort(-singular_values_algo)
singular_values_algo_sorted = singular_values_algo[idx]
print("Singular values (sorted descending):", singular_values_algo_sorted)
```

---

# 2. NumPy SVD (LAPACK Implementation)
NumPy computes the SVD using a stable algorithm (bidiagonalization + QR/divide‑and‑conquer) and **always sorts singular values**.

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print("NumPy SVD singular values (always sorted):", S)
print("U matrix:", U)
print("V^T matrix:", Vt)
```

---

# 3. Comparison
```python
print("Comparison (algorithm vs NumPy):")
print("Manual (sorted):", singular_values_algo_sorted)
print("NumPy:          ", S)
```

---

# ✔️ Summary
- Singular values come from **eigenvalues of \(X^T X\)**.  
- The eigenvalue solver **does not guarantee ordering**.  
- SVD libraries (LAPACK, NumPy, MATLAB, PyTorch) explicitly **sort the singular values** and reorder \(U\) and \(V\) accordingly.  


