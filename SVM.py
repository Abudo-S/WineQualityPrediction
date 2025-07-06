import numpy as np
import matplotlib.pyplot as plt
from Kernel import Kernel
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score

'''
A higher value of λ increases the cost of misclassifications (shrinking the weights), forcing the algorithm to try harder to classify all training points correctly, 
even if it means a smaller margin. This can lead to a smaller margin and potentially overfitting if λ is too large.  
A lower value of λ decreases the cost of misclassifications (expanding the weights), allowing the algorithm to have a larger margin, 
even if it misclassifies some training points. This can lead to underfitting if λ is too small.
Generally λ needs to be added if we start noting some overfitting or inadeguate low test performance.

The learning rate aims to adjust the weights and bias (determines how big of a step the algorithm takes in the direction opposite to the gradient)
n_iterations is the number of epochs needed to converge to the near-optimal weights and bias, reducing consequently the loss.
'''
class SVM:
    #the hyperparameters need to be tuned by k-CV
    '''
    we can choose either batch gradient descent "BGD" or stocastic gradient descent "SGD"
    Note that SGD is more sensible to the noise since it applies weight/bias updates based on each data point
    '''
    def __init__(self, gradient_strategy = "BGD", learning_rate = 0.001, lambda_param = 0.01, n_iterations = 1000, kernel:Kernel=None):
        self.gradient_strategy = gradient_strategy
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.kernel = kernel

        #for performance calculation
        self._train_losses = []
        self._test_losses = []
        self._train_accuracies = []
        self._test_accuracies = []

    def fit(self, X:np.ndarray, y:np.ndarray, X_validation:np.ndarray=None, y_validation:np.ndarray=None):
        y_ = np.where(y > 0, 1, -1)
        y_validation = np.where(y_validation > 0, 1, -1)

        if self.kernel is not None:
            self._fit_ksvm(X, y_)
        else:
            self._fit_svm(X, y_, X_validation, y_validation)
        

    def _fit_svm(self, X, y_, X_validation, y_validation):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Batch gradient descent
        if(self.gradient_strategy == "BGD"):
            for epoch in range(self.n_iterations):
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
                
                #calculate performance metrices for current epoch
                self._record_metrics(X, y_, X_validation, y_validation, epoch)

        else: #Stocastic gradient descent
            X_indices = np.arange(n_samples)

            for epoch in range(self.n_iterations):
                np.random.shuffle(X_indices)
                X_per_epoch = X[X_indices]

                for idx, x_i in enumerate(X_per_epoch):
                    margin_score = y_[idx] * (np.dot(x_i, self.weights) + self.bias) #np.dot(wi * xi) = sum(wi * xi)
                    
                    #note that the stepUpdate is in the opposite of weight -= (since we're using gradient descent)
                    if margin_score < 1:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                        self.bias -= self.learning_rate * y_[idx]
                    else:
                        self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)

                #calculate performance metrices for current epoch
                self._record_metrics(X_per_epoch, y_, X_validation, y_validation, epoch)

    def _fit_ksvm(self, X, y_):
        n_samples = X.shape[0]
        
        #q, p parameters using K_matrix and y_
        P = matrix(np.outer(y_, y_) * self.kernel.K_matrix) #y_i * y_j * K(x_i, x_j) part
        q = matrix(np.ones(n_samples) * -1)

        #constraints for 0 <= alpha_i <= C
        G_ineq = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h_ineq = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.lambda_param)))

        #constraint for sum(alpha_i * y_i) = 0
        A_eq = matrix(y_, (1, n_samples), 'd')
        b_eq = matrix(0.0)

        #qp solver
        solution = solvers.qp(P, q, G_ineq, h_ineq, A_eq, b_eq)
        
        #alpha vector
        self.alphas = np.array(solution['x']).flatten() 
    
        #identify support vectors (SV)
        ALPHA_THRESHOLD = 1e-5
        sv_indices = self.alphas > ALPHA_THRESHOLD #points with non-zero alpha are SVs
        self.sv_data_points = X[sv_indices]
        self.sv_y = y_[sv_indices]
        self.sv_alphas = self.alphas[sv_indices]
        
        #use a support vector w.r.t. lambda_param
        margin_sv_indices = (self.alphas > ALPHA_THRESHOLD) & (self.alphas < self.lambda_param - ALPHA_THRESHOLD)
        
        if np.sum(margin_sv_indices) > 0:
            bias_values = []
            for i in np.where(margin_sv_indices)[0]:
                #calculate f(x_i) = sum(alpha_j * y_j * K(x_j, x_i)) for the current SV x_i
                f_x_i = np.sum(self.alphas[sv_indices] * y_[sv_indices] * self.kernel.K_matrix[sv_indices, i])
                bias_values.append(y_[i] - f_x_i)
                
            self.bias = np.mean(bias_values)
        elif np.sum(sv_indices) > 0: #fallback to the first alpha SV if there's no margin alpha SVs
            i = np.where(sv_indices)[0][0]
            #calculate f(x_i) = sum(alpha_j * y_j * K(x_j, x_i)) for the current SV x_i
            f_x_i = np.sum(self.alphas[sv_indices] * y_[sv_indices] * self.kernel.K_matrix[sv_indices, i])

            self.bias = y_[i] - f_x_i
        else:
            self.bias = 0.0

    def predict(self, X:np.ndarray):
        predictions = np.zeros(X.shape[0])

        if self.kernel is not None:
            K_test = self.kernel.compute_kernel_matrix(X, self.sv_data_points)

            predictions = np.dot(K_test, self.sv_alphas * self.sv_y) + self.bias
        else:
            predictions = np.dot(X, self.weights) + self.bias #linear prediction

        return np.sign(predictions) #y
    
    '''
    loss = mean(max(0, 1 - y * f(x))) + λ * ||w||^2
    '''
    def _hinge_loss(self, X, y_true):
        scores = np.dot(X, self.weights) + self.bias
    
        #average hinge loss for X datapoints
        avg_hinge_loss = np.mean(np.maximum(0, 1 - y_true * scores))
        
        #consider L2 regularization: lambda * ||w||^2
        regularization_term = self.lambda_param * np.dot(self.weights, self.weights)

        return avg_hinge_loss + regularization_term

    def _record_metrics(self, X_train, y_train, X_test, y_test, epoch):
        #training loss and accuracy
        train_loss = self._hinge_loss(X_train, y_train)
        self._train_losses.append(train_loss)

        train_preds = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        self._train_accuracies.append(train_acc)

        #validation loss and accuracy
        if X_test is not None and y_test is not None:
            test_loss = self._hinge_loss(X_test, y_test)
            self._test_losses.append(test_loss)
        
            test_preds = self.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            self._test_accuracies.append(test_acc)

    def get_model_metrics_evaluation(self):
        return {'loss': self._train_losses,
                'val_loss': self._test_losses,
                'accuracy': self._train_accuracies,
                'val_accuracy': self._test_accuracies}

    '''
    Standard visualization [not kernerlized]
    '''
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