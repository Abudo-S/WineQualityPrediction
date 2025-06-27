import numpy as np
import matplotlib.pyplot as plt


'''
It squashes any real-valued number into a value between 0 and 1, which can be interpreted as a probability. σ(z)= 1 / (1+e^−z)
it seeks to find a logaritmic equation(sigmoid) that best describes how one or more independent variables (features)
relate to a dependent variable (target label).
The model aims to find the "best-fit" approximated hyperplane that MINIMIZES the gradient(not exact value predictor "based on a predefined threshold")

The learning rate aims to adjust the weights and bias (determines how big of a step the algorithm takes in the direction opposite to the gradient)
n_iterations is the number of epochs needed to converge to the near-optimal weights and bias, reducing consequently the loss.
'''
class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.bias = None
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, -1)
        
        for _ in range(self.n_iterations):
            y_predicted = self._predict_prob(X)

            first_derivative_dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y_))
            second_derivative_db = (1 / n_sample) * np.sum(y_predicted - y_)

            #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
            self.weights -= self.learning_rate * first_derivative_dw #adjust weights with respect to the learning rate
            self.bias -= self.learning_rate * second_derivative_db #adjust bias with respect to the learning rate

    def predict(self, X: np.ndarray):
        y_predicted = self._predict_prob(X)
        
        #converting probabilities to classes based on a single threshold, make LR considered as linear model, despite the expliot of sigmoid
        return np.where(y_predicted >= self.threshold, 1, -1) 

    def _predict_prob(self, X):
        linear_y = np.dot(X, self.weights) + self.bias
        
        return self._sigmoid(linear_y)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def visualize_LR(self, X, y):
        plt.clf()

        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', s=20)

        #bounary start point
        x0_1 = np.amin(X[:, 0]) #x
        x0_2 = np.amax(X[:, 0]) #y

        #boundary end point
        x1_1 = get_hyperplane_value(x0_1, self.weights, self.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, self.weights, self.bias, 0)

        plt.plot([x0_1, x0_2], [x1_1, x1_2], color='green', linestyle='-', linewidth=2, label='LR decision boundary')
        plt.legend()

        plt.show()

