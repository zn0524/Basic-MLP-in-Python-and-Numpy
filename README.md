# Basic MLP From Scratch

A basic Multi Layer Perceptron I made from scratch using only NumPy.
No TensorFlow or PyTorch was used for the actual neural network implementation. The
goal of this project was mainly to understand how neural networks actually work under
the hood instead of just importing layers from a framework.
This project includes:
* forward propagation
* backpropagation
* gradient descent
* sigmoid activation
* manual weight updates
* binary classification
The model is trained on the Iris dataset using only petal length and petal width.

# Dataset

Using the Iris dataset from sklearn:

```python
from sklearn import datasets
#loading iris dataset
iris = datasets.load_iris()
X = iris["data"][:, (2,3)] #input data
y = (iris["target"] == 2).astype(int)
y = y.reshape([150,1]) #truth values
```

The network predicts whether the flower belongs to class 2 or not.

# Architecture

The network architecture is:
2 Inputs
↓
4 Hidden Neurons
↓
1 Output Neuron
Created with:
```python
python
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
```

# Weight Initialization

Weights are initialized randomly:
```python
self.weights1 = np.random.randn(self.input_size, self.hidden_size)
self.weights2 = np.random.randn(self.hidden_size, self.output_size)
```
Biases are initialized as zeros:

```python
self.bias1 = np.zeros((1, hidden_size))
self.bias2 = np.zeros((1, output_size))
```

# Feed Forward

Hidden layer calculation:
```python
layer1 = X.dot(self.weights1)+self.bias
activation1 = sigmoid(layer1)
```
This line:
```python
layer1 = X.dot(self.weights1)+self.bias
```
matches the equation:

$$
z = XW + b
$$

Where:
* (X) = input data
* (W) = weights
* (b) = bias
* (z) = weighted sum before activation
Then this line:
```python
activation1 = sigmoid(layer1)
```
applies the sigmoid activation function:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

which compresses values between 0 and 1.


# Output Layer
```python
layer2 = activation1.dot(self.weights2) + self.bias
activation2 = sigmoid(layer2)
```
This again follows:

$$
z=XW+b
$$

except now:

* (X) is the hidden layer activations
* (W) is weights

Then sigmoid is applied again to create the final prediction probability.

# Error Calculation

```python
error = activation2 - y
```
This calculates the difference between:

* predicted values
* actual labels

Which is basically:

$$
error=\hat{y}-y
$$

Where:
* $(\hat{y})$ = prediction
* (y) = actual value


# Backpropagation

Output layer gradients:
```python
d_weights2 = activation1.T.dot(error * sigmoid_derivative(layer2))
```
This computes:

$$
\frac{\partial L}{\partial W_2}=a_1^T(\hat{y}-y)\sigma'(z_2)
$$

Where:
* $(a_1)$ = hidden layer activations
* $(W_2)$ = second layer weights
* $(\sigma'(z))$ = sigmoid derivative
Hidden layer error:
```python
error_hidden = error.dot(self.weights2.T) * sigmoid_derivative(layer1)
```
This propagates the error backwards through the network using the chain rule.
Equation:

$$
\delta_1=(\delta_2W_2^T)\sigma'(z_1)
$$

# Sigmoid Derivative

```python
def sigmoid_derivative(z):
  s = sigmoid(z)
  return s * (1-s)
```
This matches:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

This derivative is needed for backpropagation.


# Weight Updates

```python
self.weights2 -= self.learning_rate * d_weights
self.weights1 -= self.learning_rate * d_weights
```
This follows the gradient descent equation:

$$
w:=w-\eta\frac{\partial L}{\partial w}
$$

Where:
* $(w)$ = weights
* $(\eta)$ = learning rate
* $(\frac{\partial L}{\partial w})$ = gradient
The network slowly adjusts the weights to minimize error.

# Training the model:
```python
mlp.fit(X, y)
```
Prediction:
```python
y_pred = mlp.predict(X)
```
Accuracy:
```python
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.2f}")
```
# Future Improvements

Some things I might add later:

* multiple hidden layers
* ReLU activation
* softmax output
* better loss functions
* mini batch training
* visualization tools
* multiclass classification support

# Educational Purpose

This project was built mainly for learning and experimentation.

