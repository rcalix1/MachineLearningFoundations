# Web Perceptron

* Implement the Perceptron on a website using either JavaScript or PyScript with Python
* link

## A visual example

![perceptron](websitePerceptronHW.png)

## PyScript Examples

* https://github.com/rcalix1/MachineLearningFoundations/tree/main/DeployToWeb
* https://rcalix1.github.io/Build-Fun-AI-Projects-that-Run-on-the-Web/volume-1-pyscript-and-knn/chapter3/AppKNNiris/index2.html
* https://github.com/rcalix1/Build-Fun-AI-Projects-that-Run-on-the-Web/tree/main/volume-1-pyscript-and-knn/chapter3/AppKNNiris
* https://pyscript.com/@examples

## Python Code

* https://github.com/rcalix1/MachineLearningFoundations/blob/main/2025/InClassPerceptron.ipynb
* https://github.com/rcalix1/MachineLearningFoundations/tree/main/NeuralNets/perceptron
* 

## Code

```

import numpy as np

# -------------------------------------------------
# Example trained perceptron weights from Python
# -------------------------------------------------

weights = np.array([
    12,
    -6,
    3,
    -9
])

bias = 4


# -------------------------------------------------
# STEP 1:
# Normalize weights to [-1, 1]
# -------------------------------------------------

max_abs = max(
    np.max(np.abs(weights)),
    abs(bias)
)

weights_norm = weights / max_abs
bias_norm = bias / max_abs


# -------------------------------------------------
# STEP 2:
# Map [-1, 1] --> [0, 100]
#
# 0   = maximum negative
# 50  = zero
# 100 = maximum positive
# -------------------------------------------------

weights_pots = 50 * (weights_norm + 1)
bias_pot = 50 * (bias_norm + 1)


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("Original weights:")
print(weights)

print("\nNormalized weights:")
print(weights_norm)

print("\nPotentiometer settings:")
print(weights_pots)

print("\nBias potentiometer:")
print(bias_pot)

```

and results 

```

Original weights:
[12 -6  3 -9]

Normalized weights:
[ 1.   -0.5   0.25 -0.75]

Potentiometer settings:
[100.   25.   62.5  12.5]

Bias potentiometer:
66.67



```

## “Liminal” = threshold between states

* The Liminal Engine
* 






