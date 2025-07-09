import numpy as np

class GenerationPoint:
    
    def generer_donnees_separables(n_points=100, noise=0.1):
        """
        Génère deux classes de points linéairement séparables.
    
        - Classe 1 : points autour de (2, 2), label 1
        - Classe -1 : points autour de (-2, -2), label -1
        """
        np.random.seed(42)
    
        # Moitié des points pour chaque classe
        n_half = n_points // 2
    
        # Classe 1 : autour de (2, 2)
        class1 = np.random.randn(n_half, 2) * noise + np.array([2, 2])
        labels1 = np.ones(n_half)
    
        # Classe -1 : autour de (-2, -2)
        class2 = np.random.randn(n_points - n_half, 2) * noise + np.array([-2, -2])
        labels2 = -np.ones(n_points - n_half)
    
        # Fusionner les deux classes
        X = np.vstack((class1, class2))
        y = np.hstack((labels1, labels2))
    
        # Mélanger les points
        indices = np.random.permutation(n_points)
        return X[indices], y[indices]
