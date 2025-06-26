import numpy as np
from sklearn.metrics import accuracy_score

'''
A simple algorithm for hyperparameter tuning, it evaluates a model with fixed hyperparameters.
it outputs the scores of validation folds after training on (k-1) folds using the fixed hyperparameters.
it can be used also to evaluate various combinations of params though the mean of validation-scores/comb.
'''
class KCrossValidation:
    def __init__(self, n_folds=5, model_class=None, evaluation_metric=accuracy_score):
        self.n_folds = n_folds
        self.evaluation_metric = evaluation_metric
        self.model_class = model_class

    '''
    Returns a list of scores, one for each validation fold.
    '''
    def evaluate_single_params(self, X:np.ndarray, y:np.ndarray, model_params=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices) # shuffle the indices for randomness

        # determine fold sizes
        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds, dtype=int)
        # distribute remainder samples to the first 'n_samples % self.n_folds' folds
        fold_sizes[:n_samples % self.n_folds] += 1
        
        # fill fold with indices
        current = 0
        folds_indices = []
        for fold_size in fold_sizes:
            folds_indices.append(indices[current : current + fold_size])
            current += fold_size
        
        scores = []
        params_str = ','.join([f'{k}:{v}' for k,v in model_params.items()])
        # iterate through k folds
        for i in range(self.n_folds):
            val_indices = folds_indices[i]
            
            # Other k-1 folds are for training
            train_indices_list = [folds_indices[j] for j in range(self.n_folds) if j != i]
            train_indices = np.concatenate(train_indices_list)

            # split data into training and validation sets for i-th fold
            X_train_fold, y_train_fold = X[train_indices], y[train_indices]
            X_val_fold, y_val_fold = X[val_indices], y[val_indices]

            # initialize the model using our params
            if model_params is not None:
                model = self.model_class(**model_params)
            else:
                model = self.model_class()

            model.fit(X_train_fold, y_train_fold)

            # evaluate the used params on the validation fold
            y_pred = model.predict(X_val_fold)
            score = self.evaluation_metric(y_val_fold, y_pred)
            scores.append(score)
            
            print(f"Fold {i+1}/{self.n_folds} for params [{params_str}] - Score: {score:.4f}")

        return scores
    
    '''
    Returns a list of mean scores, one for each params comb evaluation.
    model_params is an array of dict
    '''
    def evaluate_params_combinations(self, X:np.ndarray, y:np.ndarray, model_params_list=None):
        mean_scores_per_comb = list()

        for model_params in model_params_list:
            mean_scores_per_comb.append((model_params, np.mean(self.evaluate_single_params(X, y, model_params))))

        return mean_scores_per_comb