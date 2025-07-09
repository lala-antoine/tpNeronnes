import numpy as np
from couche_neurones import CoucheNeurones

class PerceptronMultiCouches:
    def __init__(self, architecture, learning_rate=0.01, activation='sigmoid'):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.activation = activation
        self.couches = []

        for i in range(len(architecture) - 1):
            act = activation
            if i == len(architecture) - 2:
                act = 'sigmoid'  # dernière couche : sigmoïde (binaire)
            self.couches.append(
                CoucheNeurones(architecture[i], architecture[i+1], activation=act, learning_rate=learning_rate)
            )

        self.history = {'loss': [], 'accuracy': []}

    def forward(self, X):
        # X shape = (n_samples, n_features)
        output = X.T
        for couche in self.couches:
            output = couche.forward(output)
        return output.T  # (n_samples, n_output)

    def compute_loss(self, y_true, y_pred):
        # Erreur quadratique moyenne
        return np.mean((y_true - y_pred) ** 2)

    def compute_accuracy(self, y_true, y_pred):
        preds = (y_pred > 0.5).astype(int)
        return np.mean(preds.flatten() == y_true.flatten())

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        # Gradient initial (dérivée du MSE)
        grad_loss = 2 * (y_pred.T - y_true.T) / m  # shape (n_output, n_samples)
        grad = grad_loss

        # Propagation arrière couche par couche
        for couche in reversed(self.couches):
            grad = couche.backward(grad)

    def train_epoch(self, X, y):
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        self.backward(X, y, y_pred)
        accuracy = self.compute_accuracy(y, y_pred)

        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        return loss, accuracy

    def fit(self, X, y, epochs=100, verbose=True):
        for epoch in range(epochs):
            loss, acc = self.train_epoch(X, y)
            if verbose and epoch % 10 == 0:
                print(f"Époque {epoch:3d} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    def predict(self, X):
        return self.forward(X)
