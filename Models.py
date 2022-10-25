import numpy as np
from scipy import linalg

class Model:
    def __init__(self, model_type, lamb=None):
        self.supported_models = {'Logistic', 'Logistic Lasso', 'SVC', 'SVC_C', 'LDA'}
        self.requires_gradient_descent = {'Logistic', 'Logistic Lasso', 'SVC', 'SVC_C'}
        
        if model_type == 'Logistic':
            self.model = Logistic(None, None)
        elif model_type == 'Logistic Lasso':
            self.model = Logistic('Lasso', lamb)
        elif model_type == 'SVC':
            self.model = SVC(lamb, False)
        elif model_type == 'SVC_C':
            self.model = SVC(lamb, True)
        elif model_type == 'LDA':
            self.model = LDA()
        else:
            raise ValueError(f'Unsupported model type provided. Supported model types: {self.supported_models.keys()}')
        
        self.model_type = model_type
        self.coef_ = []
    
    # Optionally pass in gradient descent parameters or use defaults (see GradientDescent class)
    def fit(self, X, Y, SVC_Y_transform=True, LDA_X_drop_B0=True, adaptive_descent=False, initial_B=None, max_iterations=None, 
            etas=None, tol=None, err=None, show_iter=True):
        if self.model_type in self.requires_gradient_descent:
            computer = GradientDescent(adaptive=adaptive_descent)

            # If Class labels are 0/1 SVC requires -1/1
            if self.model_type in ['SVC', 'SVC_C'] and SVC_Y_transform:
                Y = self.model.Y_transform(Y)
            # Run the descent and check convergence
            self.coef_, converged = computer.descend(self.model.gradient, X, Y, initial_B, 
                                                                  max_iterations, tol, etas, err, show_iter)
            # TODO: Decide how to handle what happens when descent doesn't converge
        elif self.model_type == 'LDA':
            if LDA_X_drop_B0:
                X = self.model.X_transform(X)
            self.coef_ = self.model.closed_form(X, Y)
        else:
            raise ValueError('Tried to fit a model with an unknown type.')
    
    def predict(self, X, LDA_X_drop_B0=True, *args):
        if self.model_type == 'LDA' and LDA_X_drop_B0:
            X = self.model.X_transform(X)
        return self.model.predict(X, self.coef_, *args)

class Logistic:
    def __init__(self, type=None, lamb=None):
        self.gradient = None
        self.lamb = None
        if type is None:
            self.gradient = self.logistic_gradient
        elif lamb is None:
            raise ValueError(f'No lambda value was provided for penalized logistic regression.')
        else:
            self.lamb = lamb
            if type == 'Lasso':
                self.gradient = self.logistic_lasso_gradient
            else:
                raise ValueError(f'Unsupport type for penalized logistic regression.')

    # Logistic Regression Gradient
    def logistic_gradient(self, X, Y, B):
        half =  np.exp(X @ B)
        p = half / (1 + half)
        gradient = -1 * (Y - p).T @ X
        
        return gradient

    # Logistic Regression with Lasso Penalty
    def logistic_lasso_gradient(self, X, Y, B):
        half =  np.exp(X @ B)
        p = half / (1 + half)
        partial_gradient = -1 * (Y - p).T @ X
        
        # Add on +/- lambda * B to the gradient
        lamb_beta = np.nan_to_num(self.lamb * -1 * (B / (B * -1)))
        gradient = partial_gradient + lamb_beta
        
        return gradient
    
    def predict(self, X, B, return_prob=False, prob_cutoff=.5):
        probabilities = 1 / (1 + np.exp(-1 * X @ B))
        return probabilities if return_prob else np.array(probabilities >= prob_cutoff).astype(int)

class SVC:
    def __init__(self, lamb=None, reverse_lambda=False):
        if lamb is None:
            raise ValueError('A lambda must be provided to SVC.')
        self.lamb = None
        self.C = None
        if reverse_lambda:
            self.gradient = self.SVC_C_gradient
            self.C = lamb
        else:
            self.gradient = self.SVC_gradient
            self.lamb = lamb

    def Y_transform(self, Y):
        # GRADIENT CALCULATION REQUIRES CLASS LABELS 1 / -1
        return (Y - 1) + Y

    # Implements SVC via some matrix magic. This requires 0/1 class labels and converts to 1/-1
    def SVC_gradient(self, X, Y, B):        
        # I know this looks crazy but trust it works :)
        mask = (1 - (Y * (X @ B))) <= 0

        if_not_0_replace_w = (self.lamb * 2 * B.sum()) - (X.T * Y).T
        return (2 * self.lamb * B.sum() * mask.sum() + if_not_0_replace_w[~mask].sum(axis=0)) / len(Y)

    # This uses lambda to minimize the loss due to bad observations and not Big betas. This should be equivalent
    # regular lambda ~ (1/2) * C
    def SVC_C_gradient(self, X, Y, B):
        # I know this looks crazy but trust it works :)
        mask = (1 - (Y * (X @ B))) <= 0

        if_not_0_replace_w = B - (self.C * (X.T * Y).T)
        return (B * mask.sum() + if_not_0_replace_w[~mask].sum(axis=0)) / len(Y)
    
    def predict(self, X, B, cutoff=0):
        return np.array(X @ B >= cutoff).astype(int)

class LDA:
    def __init__(self):
        self.closed_form = self.LDA_closed_form

    # NO B0 FOR LDA
    def X_transform(self, X):
        return X.T[1:].T

    def LDA_closed_form(self, X, Y):
        X_group1 = X[Y == 1]
        X_group2 = X[Y == 0]
        
        S1 = np.cov(X_group1.T)
        S2 = np.cov(X_group2.T)

        n1 = len(X_group1)
        n2 = len(X_group2)
        x_bar_1_2 = X_group1.mean(axis=0) - X_group2.mean(axis=0)
        
        Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        
        # if p == 1
        if len(Sp.shape) != 0:
            a = linalg.inv(Sp) @ x_bar_1_2
        else:
            a = (1 / Sp) * x_bar_1_2

        
        z_bar1 = (a @ X_group1.T).mean()
        z_bar2 = (a @ X_group2.T).mean()

        cutoff = .5 * (z_bar1 + z_bar2) - np.log(n1 / n2)
        
        # a * X >= cutoff -> class 1; else class 2
        return a, cutoff

    def predict(self, X, B):
        a, cutoff = B
        mask = a @ X.T >= cutoff
        return np.where(mask, np.ones(len(X)), np.zeros(len(X)))

class GradientDescent:
    def __init__(self, adaptive=False):
        if adaptive:
            self.descend = self.adaptive_gradient_descent
        else:
            self.descend = self.gradient_descent
        self.default_max_iter = 75000
        self.default_tol = 1e-3
        self.default_etas = [.1, .01, .001, .0001, .00001, .000001]
        self.default_err = False

    def gradient_descent(self, reg_func, X, Y, initial_B=None, max_iterations=None, tol=None,
                         etas=None, err=None, show_iter=True):
        if max_iterations is None:
            max_iterations = self.default_max_iter
        if tol is None:
            tol = self.default_tol
        if etas is None or etas == []:
            etas = self.default_etas
        if err is None:
            self.default_err = False
        
        if not err:
            np.seterr(all="ignore")
        else:
            np.seterr(all="warn")
        
        for eta in etas:
            # reset
            iterations = 0
            if initial_B is not None:
                B = initial_B
            else:
                B = np.zeros(len(X[0]))
            gradient = np.zeros(len(X[0]))
            while iterations < max_iterations and np.isinf(B).sum() == 0 and \
                (iterations == 0 or (eta * (gradient ** 2)).sum() > tol):
                # calls the regression function
                gradient = reg_func(X, Y, B)
                B = B - (eta * gradient)
                iterations += 1

            if iterations < max_iterations and np.isinf(B).sum() == 0 and np.isnan(B).sum() == 0:
                if show_iter:
                    print(f'Gradient converged w/ {iterations} iterations and eta = {eta}')
                np.seterr(all="warn")
                return B, True
            if show_iter:
                print(f'Eta: {eta}; Iterations: {iterations}')
        print('GRADIENT DID NOT CONVERGE. RESULTS ARE BAD')
        np.seterr(all="warn")
        return B, False

    # The code below uses the 'Adagrad' gradient descent optimization algorithm to adapt the 
    # learning rate for each dimension. There are other versions of this that may be more effective
    # Directions as to how this works as well as other ideas: 
    # https://ruder.io/optimizing-gradient-descent/index.html#momentum
    def adaptive_gradient_descent(self, reg_func, X, Y, initial_B=None, max_iterations=None, 
                                  tol=None, etas=None, err=None, show_iter=True):
        if max_iterations is None:
            max_iterations = self.default_max_iter
        if tol is None:
            tol = self.default_tol
        if etas is None or etas == []:
            etas = self.default_etas
        if err is None:
            self.default_err = False
        
        if not err:
            np.seterr(all="ignore")
        else:
            np.seterr(all="warn")
        
        for eta in etas:
            # reset
            iterations = 0
            if initial_B is not None:
                B = initial_B
            else:
                B = np.zeros(len(X[0]))
            gradient = np.zeros(len(X[0]))
            SS_past_gradients = np.zeros(len(X[0]))
            while iterations < max_iterations and np.isinf(B).sum() == 0 and \
                (iterations == 0 or (eta * (gradient ** 2)).sum() > tol):
                # calls the regression function
                gradient = reg_func(X, Y, B)
                
                # Where SS_past_gradients is sum of squares of past gradients
                SS_past_gradients += gradient ** 2
                B = B - ((eta * gradient) / (np.sqrt(SS_past_gradients) + 1e-8))
                
                iterations += 1

            if iterations < max_iterations and np.isinf(B).sum() == 0 and np.isnan(B).sum() == 0:
                if show_iter:
                    print(f'Gradient converged w/ {iterations} iterations and eta = {eta}')
                np.seterr(all="warn")
                return B, True
            if show_iter:
                print(f'Eta: {eta}; Iterations: {iterations}')
        print('GRADIENT DID NOT CONVERGE. RESULTS ARE BAD')
        np.seterr(all="warn")
        return B, False