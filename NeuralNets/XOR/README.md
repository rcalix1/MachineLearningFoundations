## XOR

* https://github.com/rcalix1/MachineLearningFoundations/blob/main/NeuralNets/PyTorch/FoundationsOfNeuralNetworks_gradients_XOR.ipynb
* 

## Derivative

## Squared Error Gradient (Simple)

### Start

$$
L = (wx + b - y)^2
$$

---

### Rule

$$
\frac{d}{dw} \left( (\text{something})^2 \right)
= 2(\text{something}) \cdot \frac{d}{dw}(\text{something})
$$

---

### Apply to $w$

$$
\frac{dL}{dw}
= 2(wx + b - y) \cdot \frac{d}{dw}(wx + b - y)
$$

---

### Inner derivative

$$
\frac{d}{dw}(wx + b - y) = x
$$

---

### Final

$$
\frac{dL}{dw} = 2(wx + b - y)x
$$

---

### Simple idea

$$
\text{error} = (wx + b - y)
$$

$$
\text{gradient} = 2 \cdot \text{error} \cdot x
$$







## Derivative of the Perceptron

