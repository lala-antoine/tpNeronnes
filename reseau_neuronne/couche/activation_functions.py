import numpy as np

class ActivationFunction:
    def __init__(self, name='sigmoid'):
        self.name = name.lower()
        if self.name == 'sigmoid':
            self.func = self.sigmoid
            self.derivative = self.sigmoid_derivative
        elif self.name == 'tanh':
            self.func = self.tanh
            self.derivative = self.tanh_derivative
        elif self.name == 'relu':
            self.func = self.relu
            self.derivative = self.relu_derivative
        else:
            raise ValueError(f"Activation {name} not supported")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def __call__(self, x):
        return self.func(x)

    def grad(self, x):
        return self.derivative(x)
