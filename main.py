# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np


class ONN:
    def __init__(self, X, y, weightsize):
        self.X = X
        self.y = y
        self.weightsize = weightsize
        self.weights = {}
        self.weights[0] = np.random.randn(self.X.shape[1], weightsize[0]) / np.sqrt(self.X.shape[1])

        for dep in range(1, len(self.weightsize)):
            self.weights[dep] = np.random.randn(self.weightsize[dep - 1], self.weightsize[dep]) / np.sqrt(
                self.weightsize[dep - 1])
        self.weights[len(self.weightsize)] = np.random.randn(self.weightsize[len(self.weightsize) - 1], 1) / np.sqrt(
            self.weightsize[len(self.weightsize) - 1])

    def sigmoid(self, Xfactor):
        return 1 / (1 + np.exp(-Xfactor))

    def activation(self, weights, X, afunc):
        linear = np.dot(X, weights)
        if afunc == 'tanh':
            output = np.tanh(linear)
        else:
            output = self.sigmoid(linear)
        return output

    def costFunction(self, y, output):
        cost = np.mean((y - output) ** 2)
        return cost

    def gradient(self, input, y, afunc, weight):
        if afunc == 'cost':
            sigmoid_output = self.sigmoid(np.dot(input, weight))
            return np.dot(input.T, (sigmoid_output * (1 - sigmoid_output)) * (sigmoid_output - y))
        else:
            return np.dot(input.T, 1 / np.cosh(np.dot(input, weight)))

    def feedforward(self, initial, weights):
        allinputs = {}
        for key, weight in weights.items():
            if key == 0:
                allinputs[0] = initial
            elif key == len(weights) - 1:
                allinputs[key] = self.activation(weight, allinputs[key - 1], '')
            else:
                allinputs[key] = self.activation(weight, allinputs[key - 1], 'tanh')

        cost = self.costFunction(self.y, allinputs[len(weights) - 1])
        return cost, allinputs

    def backprop(self, output, alpha, gamma):
        for key in range(len(output) - 1, 0, -1):
            if key == len(output) - 1:
                grad = self.gradient(output[key - 1], self.y, 'cost', self.weights[key])
                self.weights[key] -= alpha * grad
                grad2 = grad
            else:
                grad = self.gradient(output[key - 1], self.y, '', self.weights[key])
                self.weights[key] -= alpha * grad - gamma*(grad-grad2)
                grad2 = grad
        return self.weights

    def fit(self, num_iterations=700, alpha=0.001, gamma = 0.35):
        linear = self.activation(self.weights[0], self.X, 'tanh')
        for i in range(num_iterations):
            cost, forward = self.feedforward(linear, self.weights)
            print("Cost is:", cost)

            # Consider using adaptive learning rate methods like Adam optimizer
            self.weights = self.backprop(forward, alpha, gamma)
            linear = self.activation(self.weights[0], self.X, 'tanh')

    def predict(self):
        linear = self.activation(self.weights[0], self.X, 'tanh')
        cost, allinputs = self.feedforward(linear, self.weights)
        return cost, allinputs[len(allinputs)-1]

