import numpy as np
import matplotlib.pyplot as plt

'''
A higher value of λ increases the cost of misclassifications, forcing the algorithm to try harder to classify all training points correctly, 
even if it means a smaller margin. This can lead to a smaller margin and potentially overfitting if λ is too large.  
A lower value of λ decreases the cost of misclassifications, allowing the algorithm to have a larger margin, 
even if it misclassifies some training points. This can lead to underfitting if λ is too small.

The learning rate aims to adjust the weights and bias (determines how big of a step the algorithm takes in the direction opposite to the gradient)
n_iterations is the number of epochs needed to converge to the near-optimal weights and bias, reducing consequently the loss.
'''
class SVM:
    #the hyperparameters need to be tuned by k-CV
    '''
    we can choose either batch gradient descent "BGD" or stocastic gradient descent "SGD"
    Note that SGD is more sensible to the noise since it applies weight/bias updates based on each data point
    '''
    def __init__(self, gradient_strategy = "BGD", learning_rate = 0.001, lambda_param = 0.01, n_iterations = 1000):
        self.gradient_strategy = gradient_strategy
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, -1)

        #Batch gradient descent
        if(self.gradient_strategy == "BGD"):
            for _ in range(self.n_iterations):
                dw = np.zeros(n_features)
                db = 0                   

                for idx, x_i in enumerate(X):
                    margin_score = y_[idx] * (np.dot(x_i, self.weights) + self.bias)

                    if margin_score < 1:
                        dw += (-y_[idx] * x_i)
                        db += (-y_[idx])

                dw_avg = dw / n_samples
                db_avg = db / n_samples
                
                #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                self.weights -= self.learning_rate * (dw_avg + (2 * self.lambda_param * self.weights))
                self.bias -= self.learning_rate * db_avg

        else: #Stocastic gradient descent
            for _ in range(self.n_iterations):
                for idx, x_i in enumerate(X):
                    margin_score = y_[idx] * (np.dot(x_i, self.weights) + self.bias) #np.dot(wi * xi) = sum(wi * xi)
                    
                    #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                    if margin_score < 1:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                        self.bias -= self.learning_rate * y_[idx]
                    else:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

    def visualize_svm(self, X, y):
        plt.clf()
        
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y) #for simplicity of visualization, we'll consider the first column as x, and the second column as y 

        #bounary start point
        x0_1 = np.amin(X[:, 0]) #x
        x0_2 = np.amax(X[:, 0]) #y

        #boundary end point(hyperplane)
        x1_1 = get_hyperplane_value(x0_1, self.weights, self.bias, 0)
        x1_2 = get_hyperplane_value(x0_2, self.weights, self.bias, 0)

        #boundary end point(negative margin)
        x1_1_m = get_hyperplane_value(x0_1, self.weights, self.bias, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.weights, self.bias, -1)

        #boundary end point(positive margin)
        x1_1_p = get_hyperplane_value(x0_1, self.weights, self.bias, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.weights, self.bias, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--", label = "SVM hyperplane")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.legend()

        plt.show()

