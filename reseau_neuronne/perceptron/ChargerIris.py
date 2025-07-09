from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

def charger_donnees_iris_binaire():
    """
    Charge le dataset Iris en version binaire (2 classes) pour commencer
    (Setosa vs Versicolor uniquement)
    """
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # longueur des sépales et pétales
    y = iris.target

    # Filtrer pour ne garder que Setosa (0) et Versicolor (1)
    mask = y < 2
    X_binary = X[mask]
    y_binary = y[mask]

    # Convertir les étiquettes : 0 -> -1, 1 -> +1
    y_binary = 2 * y_binary - 1

    return X_binary, y_binary

def charger_donnees_iris_complete():
    """
    Charge le dataset Iris complet avec les 3 classes
    (Setosa, Versicolor, Virginica)
    """
    iris = load_iris()
    X = iris.data[:, [0, 2]]  # longueur des sépales et pétales
    y = iris.target
    target_names = iris.target_names
    return X, y, target_names

def visualiser_iris(X, y, target_names=None, title="Dataset Iris"):
    """
    Visualise le dataset Iris avec ses différentes classes
    """
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    markers = ['*', '+', 'o']

    for i in range(len(np.unique(y))):
        mask = (y == i)
        label = target_names[i] if target_names is not None else f'Classe {i}'
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[i], marker=markers[i], s=100,
                    label=label, alpha=0.7)

    plt.xlabel('Longueur des sépales (cm)')
    plt.ylabel('Longueur des pétales (cm)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
