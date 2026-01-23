import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases for NN[2,4,1]
        self.weights = [
            np.identity(2),
            np.random.randn(4, 2), 
            np.random.randn(1, 4)
        ]
        
        self.bias = [
            np.zeros(2),
            np.random.randn(4),
            np.random.randn(1)
        ]
        
        self.z = [None] * len(self.weights)
        self.a = [None] * len(self.weights)
        self.delta = [None] * len(self.weights)
        self.learning_rate = 0.1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward_prop(self, x):
        self.z[0] = np.dot(self.weights[0], x) + self.bias[0].reshape(-1, 1)
        self.a[0] = self.z[0]

        for i in range(1, len(self.weights)):
            self.z[i] = np.dot(self.weights[i], self.a[i-1]) + self.bias[i].reshape(-1, 1)
            self.a[i] = self.sigmoid(self.z[i])

    def cost(self, batch_input):
        self.forward_prop(batch_input)
        
        target = np.sign(batch_input[0] * batch_input[1])
        prediction = self.a[-1].flatten() 
        
        total_cost = np.sum(np.square(target - prediction))
        return total_cost

    def backprop(self, batch_input):
        self.forward_prop(batch_input) 
        
        # 1. Output Layer Delta
        target = np.sign(batch_input[0] * batch_input[1])
        # Use self.a[-1] (activations) for derivative, not z
        self.delta[-1] = 2 * (self.a[-1] - target) * self.sigmoid_derivative(self.a[-1])

        # 2. Hidden Layers Delta (Propagate backwards to 0)
        for l in range(len(self.weights) - 2, -1, -1):
            # Use self.a[l] for derivative
            self.delta[l] = np.dot(self.weights[l+1].T, self.delta[l+1]) * self.sigmoid_derivative(self.a[l])     
       
        # 3. Update Weights and Biases
        for l in range(len(self.weights)):
            # Determine input to this layer
            if l == 0:
                layer_input = batch_input
            else:
                layer_input = self.a[l-1]
            
            # Gradient Descent: Subtract the gradient
            self.weights[l] -= self.learning_rate * np.dot(self.delta[l], layer_input.T)
            self.bias[l] -= self.learning_rate * np.sum(self.delta[l], axis=1)

if __name__ == "__main__":
    batch_input = np.random.uniform(-1, 1, size=(2, 100))
    
    nn = NeuralNetwork()
    
    for i in range(5000):
        nn.backprop(batch_input)
    
    print(f"Cost: {nn.cost(batch_input)}")