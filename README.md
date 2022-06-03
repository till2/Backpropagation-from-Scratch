# Backpropagation from Scratch
## _This is a toy MLP with one hidden layer for backprop._
The target vector y is fixed to [10,-2] for simplicity but can be easily modified to depend on the input vector x.

- ✨No torch or TF,  only numpy✨

## Files

| File | Desc |
| ------ | ------ |
| Backprop from Scratch.ipynb | Notebook → Download and modify the code! :) |
| Backprop from Scratch.pdf  | Rendered pdf |

## Weight derivation

<img src="https://render.githubusercontent.com/render/math?math={x}">

xfun::embed_file("eq1.pdf")

[Code for W1 Gradient](#gradient-calculations-for-weights-in-code)

LATEX2 HERE
[Code for W2 Gradient](#gradient-calculations-for-weights-in-code)

## Gradient Calculations for Weights in Code

### Gradient of Weight Matrix W⁽¹⁾

```py
h = sigmoid(np.dot(W1, x))
dL_dW2 = np.dot((-y + out), h.T)
```

### Gradient of Weight Matrix W⁽²⁾

```py
z = np.dot(W1, x)
a = (-y + out).T
b = np.dot(a, W2)
c = sigmoid(z) * (1 - sigmoid(z))
d = b.T * c
dL_dW1 = d * x.T
```
