# Week 1 - Neural Networks

## Neural network layer

A simple visualization of a deep learning neural network with an input ($x$) and a hidden layer and an output($\vec{a}^{[2]}$) with Sigmoid function as activation of all layers.

$$
x \overset{\vec{x}}{\Longrightarrow}
\begin{cases}
    \vec{w_1}^{[1]}, \vec{b_1}^{[1]} \quad a_1^{[1]} = g(\vec{w_1}^{[1]} \cdot \vec{x} + \vec{b_1}^{[1]}) \\
    \vec{w_2}^{[1]}, \vec{b_2}^{[1]} \quad a_2^{[1]} = g(\vec{w_2}^{[1]} \cdot \vec{x} + \vec{b_2}^{[1]}) \\
    \vec{w_3}^{[1]}, \vec{b_3}^{[1]} \quad a_3^{[1]} = g(\vec{w_3}^{[1]} \cdot \vec{x} + \vec{b_3}^{[1]})
\end{cases}
\enspace \overset{\vec{a}^{[1]}}{\Longrightarrow}
\begin{cases}
    \vec{w_1}^{[2]}, \vec{b_1}^{[2]} \quad a_1^{[2]} = g(\vec{w_1}^{[2]} \cdot \vec{a}^{[1]} + \vec{b_1}^{[2]})
    \enspace \overset{\vec{a}^{[2]}}{\Longrightarrow} [1, 0]
\end{cases}
$$

## More complex neural network

General form: $a_j^{[l]} = g(\vec{w_j}^{[l]} \cdot \vec{a}^{[l - 1]} + \vec{b_j}^{[l]})$
Where $l$ is an arbitrary layer and $j$ is an arbitrary unit.

## Data in tensorflow

Convert Tensor object to Numpy array by

```python
a1 = [[1, 2], [2, 4]] # some data
layer2 = Dense(units=1, activation='sigmoid')
a2 = layer2(a1)

np_arr = a2.numpy() # returns the numpy array
```

## Forward propagation

_tip)_ We name matrix variables like `W` Uppercase and array variables like `b` lowercase.

Simplified implementation of dense layer:

```python
def dense(a_in, W, b):
    units = a_in.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z) # g() is defined outside of dense()
    return a_out

x = np.array([200, 17])
W = np.array([[1, -3, 5]
              [-2, 4, -6]])
b = np.array([-1, 1, 2])

dense(x, W, b)
```

## How neural networks are implemented efficiently

Vectorized version of above implementation:

```python
def dense(A_in, W, B):
    Z = np.matmul(A_in, W) # multiply A_in and W
    Z = Z + B
    A_out = g(Z)
    return A_out

X = np.array([[200, 17]])
W = np.array([[1, -3, 5]
              [-2, 4, -6]])
B = np.array([[-1, 1, 2]])
```

_tip)_ `T` property of a numpy 2D array is its transposed version. We name a transposed matrix with a `T` at the end e.g. `AT = A.T`

_tip)_ `@` will do the `np.matmul()` job e.g. `Z = AT @ W`
