import numpy as np
from perceptron_multi_couches import PerceptronMultiCouches
import matplotlib.pyplot as plt

def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = PerceptronMultiCouches(architecture=[2, 4, 1], learning_rate=0.1, activation='tanh')
    model.fit(X, y, epochs=1000, verbose=True)

    y_pred = model.predict(X)
    print("Prédictions (valeurs réelles):")
    print(y_pred.flatten())

    # Visualiser la perte
    plt.plot(model.history['loss'])
    plt.title('Courbe de perte (loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

test_xor()
