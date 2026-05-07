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



---



# The Kernel Trick and the Linear Nature of SVMs

## Intuition

One of the most elegant ideas in classical machine learning was the realization that:

> A nonlinear classifier in the original space can be represented as a linear classifier in a transformed feature space.

The key idea is:

$$
\phi(x): \mathbb{R}^n \rightarrow \mathcal{H}
$$

where:

* $x$ is the original input vector
* $\phi(x)$ is a transformed feature representation
* $\mathcal{H}$ is a higher-dimensional feature space

---

# Simple Example: Mapping from 2D to 3D

Suppose:

$$
x = [x_1, x_2]
$$

Define the feature mapping:

$$
\phi(x) =
[x_1^2, \sqrt{2}x_1x_2, x_2^2]
$$

This maps:

$$
\mathbb{R}^2 \rightarrow \mathbb{R}^3
$$

Now a linear classifier can operate on:

$$
v_3 = \phi(x)
$$

using:

$$
f(x) = w^T v_3 + b
$$

Expanding:

$$
f(x) =
w_1x_1^2
+
w_2\sqrt{2}x_1x_2
+
w_3x_2^2
+
b
$$

Notice:

* this is nonlinear in the original variables $x_1,x_2$
* but linear in the transformed coordinates $\phi(x)$

---

# The Kernel Trick

Instead of explicitly computing:

$$
\phi(x)
$$

we compute only inner products:

$$
K(x,z)=\phi(x)^T\phi(z)
$$

For the mapping above:

$$
\phi(x)^T\phi(z)
$$

becomes:

$$
x_1^2z_1^2
+
2x_1x_2z_1z_2
+
x_2^2z_2^2
$$

which simplifies to:

$$
(x_1z_1 + x_2z_2)^2
$$

Therefore:

$$
K(x,z)=(x^Tz)^2
$$

This is the polynomial kernel.

The important insight:

> We never explicitly construct the higher-dimensional vectors.

We only compute kernel evaluations.

---

# SVM Linear Classifier in Feature Space

The SVM classifier is:

$$
f(x)=w^T\phi(x)+b
$$

A major theoretical result shows:

$$
w=\sum_i \alpha_i y_i \phi(x_i)
$$

meaning:

* the hyperplane is constructed from training examples
* only support vectors matter

Substitute into the classifier:

$$
f(x)=
\left(
\sum_i \alpha_i y_i \phi(x_i)
\right)^T
\phi(x)+b
$$

Distribute the inner product:

$$
f(x)=
\sum_i
\alpha_i y_i
\phi(x_i)^T\phi(x)
+b
$$

Now replace:

$$
\phi(x_i)^T\phi(x)=K(x_i,x)
$$

Result:

$$
f(x)=
\sum_i
\alpha_i y_i K(x_i,x)
+b
$$

---

# Why This Looks Nonlinear

The classifier:

$$
f(x)=
\sum_i
\alpha_i y_i K(x_i,x)
+b
$$

looks nonlinear because:

* the kernel itself may be nonlinear
* examples include polynomial kernels and RBF kernels

However:

> The classifier is still linear in feature space.

The nonlinearity comes entirely from the mapping:

$$
\phi(x)
$$

---

# RBF Kernel

One of the most important kernels is the Radial Basis Function (RBF):

$$
K(x,z)=e^{-\gamma ||x-z||^2}
$$

This corresponds to an implicit infinite-dimensional feature space.

Using Taylor expansion:

$$
e^{2\gamma x^Tz}
================

\sum_{k=0}^{\infty}
\frac{(2\gamma x^Tz)^k}{k!}
$$

reveals that the RBF kernel contains:

* degree 0 terms
* degree 1 terms
* degree 2 polynomial interactions
* degree 3 terms
* infinitely many higher-order interactions

Thus the RBF kernel implicitly represents an infinite-dimensional feature mapping.

---

# Historical Perspective

Classical machine learning researchers strongly favored:

* convex optimization
* geometry
* linear algebra
* eigendecomposition
* SVD methods
* quadratic programming

SVMs were attractive because:

* optimization remained convex
* solutions had elegant geometric interpretation
* the classifier was still fundamentally linear
* nonlinear behavior emerged through feature mappings

This was one of the major breakthroughs before deep learning.

---

# Modern Interpretation

Kernel methods manually define:

$$
\phi(x)
$$

through mathematical kernels.

Deep learning instead learns:

$$
\phi_\theta(x)
$$

from data.

In that sense:

* SVMs use engineered feature geometry
* neural networks learn feature geometry automatically

---

# Key Takeaway

The kernel trick is fundamentally about this idea:

> You do not need explicit coordinates to define geometry.

If you can compute valid inner products:

$$
K(x,z)=\phi(x)^T\phi(z)
$$

then you can operate as if the data lived in a richer feature space without explicitly constructing that space.

That is the mathematical beauty of kernel methods.




---
