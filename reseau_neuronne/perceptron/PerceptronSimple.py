import numpy as np
import matplotlib.pyplot as plt
from ActivationFunction import ActivationFunction
from GenerationPoint import GenerationPoint as gp

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.af = ActivationFunction("heaviside")
        self.errors_ = []

    def fit(self, X, y, max_epochs=100):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0.0
        self.errors_ = []

        for epoch in range(max_epochs):
            errors = 0
            for i in range(n_samples):
                x = X[i]
                y_true = y[i]
                linear_output = np.dot(x, self.weights) + self.bias
                y_pred = self.af.apply(linear_output)
                error = y_true - y_pred

                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
            self.errors_.append(errors)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            linear_output = np.dot(x, self.weights) + self.bias
            y_pred[i] = self.af.apply(linear_output)
        return y_pred

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]):
    """
    Analyse la convergence pour différents taux d'apprentissage
    """
    plt.figure(figsize=(12, 8))
    for lr in learning_rates:
        perceptron = PerceptronSimple(learning_rate=lr)
        perceptron.fit(X, y, max_epochs=30)
        plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, label=f"eta={lr}")

    plt.xlabel('Époque')
    plt.ylabel("Nombre d'erreurs")
    plt.title("Convergence pour différents taux d'apprentissage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

# --- Fonction pour afficher la frontière de décision ---
def plot_decision_boundary(model, X, y, title=""):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=[-2, 0, 2], colors=['#FFAAAA', '#AAFFAA'], alpha=0.5)

    for label, marker, color in zip([-1, 1], ['o', 's'], ['red', 'green']):
        plt.scatter(X[y == label, 0], X[y == label, 1], c=color, marker=marker, label=f"Classe {label}")

    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True)
    plt.show()

# ====================
# === TEST SECTION ===
# ====================

if __name__ == "__main__":
    # Données pour la fonction AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])  # -1 pour False, 1 pour True

    print("=== TEST FONCTION AND ===")
    p_and = PerceptronSimple(learning_rate=0.1)
    p_and.fit(X_and, y_and)
    print("Score AND:", p_and.score(X_and, y_and))
    for x in X_and:
        print(f"Entrée: {x}, Prédit: {p_and.predict(np.array([x]))[0]}")
    plot_decision_boundary(p_and, X_and, y_and, title="Fonction AND")

    # Données pour la fonction OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([-1, 1, 1, 1])

    print("\n=== TEST FONCTION OR ===")
    p_or = PerceptronSimple(learning_rate=0.1)
    p_or.fit(X_or, y_or)
    print("Score OR:", p_or.score(X_or, y_or))
    for x in X_or:
        print(f"Entrée: {x}, Prédit: {p_or.predict(np.array([x]))[0]}")
    plot_decision_boundary(p_or, X_or, y_or, title="Fonction OR")

    # Données pour la fonction XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([-1, 1, 1, -1])

    print("\n=== TEST FONCTION XOR ===")
    p_xor = PerceptronSimple(learning_rate=0.1)
    p_xor.fit(X_xor, y_xor)
    print("Score XOR:", p_xor.score(X_xor, y_xor))
    for x in X_xor:
        print(f"Entrée: {x}, Prédit: {p_xor.predict(np.array([x]))[0]}")
    plot_decision_boundary(p_xor, X_xor, y_xor, title="Fonction XOR (non linéaire)")
    
    print("\n=== TEST NUAGE POINT ===")
    # Génération, entraînement et affichage
    X_gen, y_gen = gp.generer_donnees_separables(noise=1.5)
    p_gen = PerceptronSimple(learning_rate=0.1)
    p_gen.fit(X_gen, y_gen)
    print("Score données générées:", p_gen.score(X_gen, y_gen))
    plot_decision_boundary(p_gen, X_gen, y_gen, title="Données générées linéairement séparables")
    
    print("\n=== ANALYSE CONVERGENCE (fonction AND) ===")
    analyser_convergence(X_and, y_and)