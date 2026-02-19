# Tiny SVD Example — Direct Algorithm vs NumPy Verification
# ------------------------------------------------------------
# This file gives you two contrasting ways to compute singular values:
#   1. Direct algorithm: compute X^T X → eigenvalues → singular values
#   2. NumPy's built‑in SVD (which includes sorting logic)
# Copy/paste or run as needed.

import numpy as np

# ------------------------------------------------------------
# 1) TINY MATRIX
# ------------------------------------------------------------
X = np.array([[3.0, 1.0],
              [2.0, 2.0]])
print("Matrix X:\n", X)

# ------------------------------------------------------------
# 2) ALGORITHM VERSION: compute singular values manually
# ------------------------------------------------------------
# Step A: Compute X^T X
XtX = X.T @ X
print("\nX^T X:\n", XtX)

# Step B: Compute eigenvalues of X^T X
# Note: eigenvalues may come out in *any order*
vals, vecs = np.linalg.eig(XtX)
print("\nEigenvalues (unordered):", vals)
print("Eigenvectors:\n", vecs)

# Step C: Singular values are sqrt(eigenvalues)
singular_values_algo = np.sqrt(np.abs(vals))  # abs() is safety for tiny numerical negatives
print("\nSingular values (unordered):", singular_values_algo)

# Step D: Sort descending (exactly what SVD implementations do)
idx = np.argsort(-singular_values_algo)
singular_values_algo_sorted = singular_values_algo[idx]
print("Singular values (sorted descending):", singular_values_algo_sorted)

# ------------------------------------------------------------
# 3) BUILT-IN SVD (LAPACK) — includes sorting of singular values
# ------------------------------------------------------------
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print("\nNumPy SVD singular values (always sorted):", S)
print("U matrix:\n", U)
print("V^T matrix:\n", Vt)

# ------------------------------------------------------------
# 4) Comparison
# ------------------------------------------------------------
print("\nComparison (algorithm vs NumPy):")
print("Manual (sorted):", singular_values_algo_sorted)
print("NumPy:          ", S)
