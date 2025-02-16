import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        
        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.w1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.w2) + self.b2
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)

        hidden_error = np.dot(d_output, self.w2.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights
        self.w2 += self.lr * np.dot(self.hidden_output.T, d_output)
        self.b2 += self.lr * np.sum(d_output, axis=0)
        self.w1 += self.lr * np.dot(x.T, d_hidden)
        self.b1 += self.lr * np.sum(d_hidden, axis=0)

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X).round()  # Convert to 0 or 1

# XOR Truth Table
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR Output

# Train MLP on XOR
mlp = MLP(input_size=2, hidden_size=2, output_size=1, lr=0.1, epochs=10000)
mlp.train(X_xor, y_xor)
print("XOR MLP Predictions:", mlp.predict(X_xor))
