import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

'''
A higher value of λ increases the cost of misclassifications, forcing the algorithm to try harder to classify all training points correctly, 
even if it means a smaller margin. This can lead to a smaller margin and potentially overfitting if λ is too large.  
A lower value of λ decreases the cost of misclassifications, allowing the algorithm to have a larger margin, 
even if it misclassifies some training points. This can lead to underfitting if λ is too small.

Since the prediction = sgn(w * x + b) that supports the binary classification; in order to generalize it for N-class classification:
*One-vs-Rest (OVR) or One-vs-All (OVA):Train a separate binary classifier for each class. For a dataset with K classes, you would train K SVM classifiers
[The class corresponding to the classifier that produces the most positive decision value is chosen as the predicted class.]
*One-vs-One (OVO): Train a binary classifier for every pair of distinct classes. For K classes, this results in training K(K-1)/2 binary SVM classifiers.
[The class that receives the most votes is chosen as the predicted class]
'''
class SVM:
    #the hyperparameters need to be tuned by k-CV or through sklearn.model_selection.validation_curve()
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bais = None

    '''
    the learning phase aims to adjust the weights and bias
    '''
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bais = 0

        y_ = np.where(y > 0, 1, -1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bais) >= 1 #np.dot(wi * xi) = sum(wi * xi)
                
                #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bais -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bais
        return np.sign(linear_output)

    def visualize_svm(self):
        plt.clf()
        
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.weights, self.bais, 0)
        x1_2 = get_hyperplane_value(x0_2, self.weights, self.bais, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.weights, self.bais, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.weights, self.bais, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.weights, self.bais, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.weights, self.bais, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

