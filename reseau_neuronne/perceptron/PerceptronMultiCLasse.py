import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PerceptronSimple import PerceptronSimple  # Assure-toi que le chemin est correct
from ChargerIris import charger_donnees_iris_binaire, charger_donnees_iris_complete, visualiser_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class PerceptronMultiClasse:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.perceptrons = {}
        self.classes = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne un perceptron par classe (stratégie un-contre-tous)
        """
        self.classes = np.unique(y)

        for classe in tqdm(self.classes, desc="Entraînement des perceptrons"):
            # 1 vs All : 1 si y == classe, -1 sinon
            y_binary = np.where(y == classe, 1, -1)

            # Entraînement d’un perceptron binaire
            perceptron = PerceptronSimple(learning_rate=self.learning_rate)
            perceptron.fit(X, y_binary, max_epochs)

            # Stockage
            self.perceptrons[classe] = perceptron

    def predict(self, X):
        """Prédit la classe la plus probable pour chaque échantillon"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]

    def predict_proba(self, X):
        """Retourne les scores (pas des probabilités normalisées)"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        return scores
    
if __name__ == "__main__":

    data = load_iris()
    X = data.data[:, :2]  # On prend 2 features pour pouvoir visualiser
    y = data.target       # Labels 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\n=== TEST MULTI-CLASSE IRIS (One-vs-All) ===")
    multi = PerceptronMultiClasse(learning_rate=0.1)
    multi.fit(X_train, y_train)

    y_pred = multi.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy Iris (2 features):", accuracy)

    # Optionnel : affichage des frontières
    from matplotlib.colors import ListedColormap

    def plot_multiclass_decision_boundary(model, X, y, title="Frontière de décision"):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ['red', 'green', 'blue']

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        for i, c in enumerate(np.unique(y)):
            plt.scatter(X[y == c, 0], X[y == c, 1], c=cmap_bold[i], label=f"Classe {c}")
        plt.title(title)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_multiclass_decision_boundary(multi, X_test, y_test, title="Perceptron Multi-Classe (Iris)")
    
    # --- Version binaire ---
    X_bin, y_bin = charger_donnees_iris_binaire()
    print(f"Données binaires : {X_bin.shape[0]} échantillons, {X_bin.shape[1]} features")
    plt.title("Iris Binaire (Setosa vs Versicolor)")
    visualiser_iris(X_bin, (y_bin + 1) // 2, ["Setosa", "Versicolor"])

    # --- Version complète ---
    X_full, y_full, noms = charger_donnees_iris_complete()
    print(f"Données complètes : {X_full.shape[0]} échantillons, {len(np.unique(y_full))} classes")
    visualiser_iris(X_full, y_full, noms)
    

