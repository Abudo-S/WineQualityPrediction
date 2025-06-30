import numpy as np  

'''
Contains non-linear kernals to be used on the target model
'''
class Kernel:
    '''
    The polynomial kernel is generally used if you have some domain knowledge, deducing that the features have polynomial relationships.
    (ex. x1 = x2^2)
    '''
    def polynomial_kernel(self, x1, x2):
        degree = self.kernel_params.get('degree', 3)
        gamma = self.kernel_params.get('gamma', 1.0)
        r = self.kernel_params.get('r', 0.0)

        return (gamma * np.dot(x1, x2) + r) ** degree

    '''
    The gaussian kernel is a general-purpose kernel which can considered as the first non-linear kernel to try 
    when we suspect that the data are not linearly separable.
    It captures non-linear relationships which are not necessarily polynomial.
    '''
    def gaussian_kernel(self, x1, x2):
        gamma = self.kernel_params.get('gamma', 1.0)
        
        return np.exp(-gamma * np.sum((x1 - x2)**2))
    
    '''
    The sigmoid kernel takes two data points and calculates a similarity score based on the tanh function.
    The score reflects how similar the two data points are in a higher-dimensional space.
    It's generally used in neural networks and image processing.
    '''
    def sigmoid_kernal(self, x1, x2):
        gamma = self.kernel_params.get('gamma', 1.0)
        r = self.kernel_params.get('r', 0.0)

        return np.tanh(gamma * np.dot(x1, x2) + r)

    def __init__(self, X:np.ndarray, kernal_func=gaussian_kernel, kernel_params=dict()):
        self.kernal_func = kernal_func
        self.kernel_params = kernel_params
        self.K_matrix = self.compute_kernel_matrix(X)
 
    '''
    Computes the kernel matrix "K" where K_ij = K(x_i, x_j).
    X2 is always None in the training phase.
    In the test phase [SVM]: X2= support vectors
    In the test phase [LR]: X2= X_train
    '''
    def compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]

        K = np.zeros((n_samples_1, n_samples_2))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                K[i, j] = self.kernal_func(self, X1[i], X2[j])

        return K
