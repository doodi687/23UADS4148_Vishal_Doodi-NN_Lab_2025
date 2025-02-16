    import numpy as np

    class Perceptron:
        def __init__(self, input_size, lr=0.1, epochs=100):
            self.weights = np.random.randn(input_size + 1)  # +1 for bias
            self.lr = lr
            self.epochs = epochs

        def activation(self, x):
            return 1 if x >= 0 else 0  # Step function

        def predict(self, x):
            x = np.insert(x, 0, 1)  # Adding bias term
            return self.activation(np.dot(self.weights, x))

        def train(self, X, y):
            for _ in range(self.epochs):
                for xi, target in zip(X, y):
                    pred = self.predict(xi)
                    error = target - pred
                    self.weights += self.lr * error * np.insert(xi, 0, 1)

        def evaluate(self, X, y):
            predictions = [self.predict(xi) for xi in X]
            return predictions

    # NAND Truth Table
    X_nand = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_nand = np.array([1, 1, 1, 0])  # NAND Output

    # XOR Truth Table
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR Output

    # Train perceptron on NAND
    p_nand = Perceptron(input_size=2)
    p_nand.train(X_nand, y_nand)
    print("NAND Predictions:", p_nand.evaluate(X_nand, y_nand))

    # Train perceptron on XOR
    p_xor = Perceptron(input_size=2)
    p_xor.train(X_xor, y_xor)
    print("XOR Predictions:", p_xor.evaluate(X_xor, y_xor))  # Will fail for XOR
