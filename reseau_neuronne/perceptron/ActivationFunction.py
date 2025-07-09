import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z >= 0, 1, -1)
        elif self.name == "sigmoid":
            return 1/(1 + np.exp(-z))
        elif self.name == "tanh":
            return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        elif self.name == "relu":
            return np.where(z >= 0, z, 0)
        elif self.name == "leaky_relu":
            return np.where(z >= 0, z, self.alpha * z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            # La dérivée de Heaviside est la distribution de Dirac
            epsilon=1e-2
            return  1 / (epsilon * np.sqrt(2 * np.pi)) * np.exp(-z**2 / (2 * epsilon**2))
        elif self.name == "sigmoid":
            return  np.exp(-z) / np.power(1 + np.exp(-z), 2)
        elif self.name == "tanh":
            return  (4 / np.power(np.exp(z) + np.exp(-z), 2))
        elif self.name == "relu":
            return np.where(z > 0, 1, 0)
        elif self.name == "leaky_relu":
            return np.where(z > 0, 1, self.alpha)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
            
    
def main():
    x = np.linspace(-10, 10, 1000)
    noms_fonctions = ["heaviside", "sigmoid", "tanh", "relu", "leaky_relu"]

    for name in noms_fonctions:
        func = ActivationFunction(name)
        y = func.apply(x)
        dy = func.derivative(x)

        # Affichage
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(x, y, label=f"{name}")
        plt.title(f"{name} - Fonction d'activation")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, dy, label=f"d({name})/dx", color='orange')
        plt.title(f"{name} - Dérivée")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# Exécution si ce fichier est lancé directement
if __name__ == "__main__":
    main()
    