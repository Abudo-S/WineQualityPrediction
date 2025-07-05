import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
It squashes any real-valued number into a value between 0 and 1, which can be interpreted as a probability. σ(z)= 1 / (1+e^−z)
it seeks to find a logaritmic equation(sigmoid) that best describes how one or more independent variables (features)
relate to a dependent variable (target label).
The model aims to find the "best-fit" approximated hyperplane that MINIMIZES the gradient(not exact value predictor "based on a predefined threshold")

The learning rate aims to adjust the weights and bias (determines how big of a step the algorithm takes in the direction opposite to the gradient)
n_iterations is the number of epochs needed to converge to the near-optimal weights and bias, reducing consequently the loss.
'''
class LogisticRegression:
    def __init__(self, gradient_strategy = "BGD", learning_rate=0.001, n_iterations=1000, threshold=0.5):
        self.gradient_strategy = gradient_strategy
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.bias = None
        self.weights = None

        #for performance calculation
        self._train_losses = []
        self._test_losses = []
        self._train_accuracies = []
        self._test_accuracies = []

    def fit(self, X:np.ndarray, y:np.ndarray, X_validation:np.ndarray=None, y_validation:np.ndarray=None):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, -1)
        
        #Batch gradient descent
        if(self.gradient_strategy == "BGD"):
            for epoch in range(self.n_iterations):
                y_predicted = self._predict_prob(X)

                first_derivative_dw = np.dot(X.T, (y_predicted - y_)) / n_sample
                second_derivative_db = np.sum(y_predicted - y_) / n_sample

                #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                self.weights -= self.learning_rate * first_derivative_dw #adjust weights with respect to the learning rate
                self.bias -= self.learning_rate * second_derivative_db #adjust bias with respect to the learning rate

                #calculate performance metrices for current epoch
                self._record_metrics(X, y_, X_validation, y_validation, epoch)
        else: #Stocastic gradient descent
            X_indices = np.arange(X.shape[0])

            for epoch in range(self.n_iterations):
                np.random.shuffle(X_indices)
                X_per_epoch = X[X_indices]

                for idx, x_i in enumerate(X_per_epoch):
                    y_predicted = self._predict_prob(X)
                    
                    first_derivative_dw_i = (y_predicted - y_[idx]) * x_i
                    second_derivative_db_i = (y_predicted - y_[idx])

                    #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                    self.weights -= self.learning_rate * first_derivative_dw_i
                    self.bias -= self.learning_rate * second_derivative_db_i

                #calculate performance metrices for current epoch
                self._record_metrics(X, y_, X_validation, y_validation, epoch)

    def predict(self, X: np.ndarray):
        y_predicted = self._predict_prob(X)
        
        #converting probabilities to classes based on a single threshold, makes LR considered as linear model, despite the expliot of sigmoid
        return np.where(y_predicted >= self.threshold, 1, -1)

    def _predict_prob(self, X):
        linear_y = np.dot(X, self.weights) + self.bias
        
        return self._sigmoid(linear_y)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    '''
    loss = -mean(y * log(p) + (1-y) * log(1-p))
    '''
    def _logistic_loss(self, X, y_true):
        y_prob = self._predict_prob(X)

        #avoid log(0) by considering a small threshold
        PROB_THRESHOLD = 1e-10
        y_prob = np.clip(y_prob, PROB_THRESHOLD, 1 - PROB_THRESHOLD)
        
        average_logistic_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

        return average_logistic_loss
    
    def _record_metrics(self, X_train, y_train, X_test, y_test, epoch):
        #training loss and accuracy
        train_loss = self._logistic_loss(X_train, y_train)
        self._train_losses.append(train_loss)

        train_preds = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        self._train_accuracies.append(train_acc)

        #validation loss and accuracy
        if X_test is not None and y_test is not None:
            test_loss = self._logistic_loss(X_test, y_test)
            self._test_losses.append(test_loss)
        
            test_preds = self.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            self._test_accuracies.append(test_acc)

    def get_model_metrics_evaluation(self):
        return {'loss': self._train_losses,
                'val_loss': self._test_losses,
                'accuracy': self._train_accuracies,
                'val_accuracy': self._test_accuracies}
    
    def visualize_lr(self, X, y):
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