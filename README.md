# Backpropagation from Scratch
## _This is a toy MLP with one hidden layer for backprop._
The target vector y is fixed to [10,-2] for simplicity but can be easily modified to depend on the input vector x.

- ✨No torch or TF,  only numpy✨

## Files

| File | Desc |
| ------ | ------ |
| Backprop from Scratch.ipynb | Notebook → Download and modify the code! :) |
| Backprop from Scratch.pdf  | Rendered pdf |

## Computation-Graph
<img src="https://github.com/till2/Backpropagation-from-Scratch/blob/main/ComputationGraph.png?raw=true" width="800" height="220"/>

## Weight derivation

### W1-Matrix gradient:
<img src="https://github.com/till2/Backpropagation-from-Scratch/blob/main/eq2.png?raw=true" width="400" height="250"/>

### W2-Matrix gradient:
<img src="https://github.com/till2/Backpropagation-from-Scratch/blob/main/eq1.png?raw=true" width="600" height="450"/>

![Gradient Calculations Code](#gradient-calculations-for-weights-in-code)


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
### Result:
We can see that the backpropagation works and the correct gradients are calculated.
The network is learning and decreasing it's loss:
<img src="https://github.com/till2/Backpropagation-from-Scratch/blob/main/LearningRates.png?raw=true" width="700" height="415"/>
