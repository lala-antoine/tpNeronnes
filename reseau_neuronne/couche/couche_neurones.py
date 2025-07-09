import numpy as np
from activation_functions import ActivationFunction

class CoucheNeurones:
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate

        # Initialisation Xavier/Glorot pour éviter l'explosion/disparition des gradients
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))

        self.activation_func = ActivationFunction(activation)

        self.last_input = None
        self.last_z = None
        self.last_activation = None

    def forward(self, X):
        # X shape = (n_features, n_samples)
        self.last_input = X
        self.last_z = np.dot(self.weights, X) + self.bias  # (n_neurons, n_samples)
        self.last_activation = self.activation_func(self.last_z)
        return self.last_activation

    def backward(self, grad_output):
        grad_activation = grad_output * self.activation_func.grad(self.last_z)  # élément-wise
        grad_weights = np.dot(grad_activation, self.last_input.T) / grad_output.shape[1]  # Moyenne sur batch
        grad_bias = np.mean(grad_activation, axis=1, keepdims=True)

        grad_input = np.dot(self.weights.T, grad_activation)  # gradient à propager en arrière

        # Mise à jour des poids et biais
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input
