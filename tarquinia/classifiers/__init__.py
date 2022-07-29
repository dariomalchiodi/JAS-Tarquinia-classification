
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state


class MLP(BaseEstimator):
    def __init__(self, max_iter=int(1E5), hidden_layer_sizes=(100,),
                 activation='relu', threshold=0.5,
                 learning_rate='constant', learning_rate_init=0.001,
                 momentum=0.9, solver='adam',
                 alpha=0.0001, shuffle=True, random_state=None):
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.momentum = momentum
        self.solver = solver
        self.alpha = alpha
        self.shuffle = shuffle
        self.random_state = random_state
        self.nn_ = None

    def __repr__(self):
        return f'MLP(max_iter={self.max_iter}, ' + \
               f'hidden_layer_sizes={self.hidden_layer_sizes}, ' + \
               f"activation='{self.activation}', " + \
               f"learning_rate='{self.learning_rate}', " + \
               f'learning_rate_init={self.learning_rate_init}, ' + \
               f'momentum={self.momentum}, ' + \
               f'alpha={self.alpha}, ' + \
               f'shuffle={self.shuffle}, ' + \
               f"solver='{self.solver}')"

    def fit(self, X, y):
        check_X_y(X, y)
        self.random_state = check_random_state(self.random_state)

        self.nn_ = MLPRegressor(max_iter=self.max_iter,
                           hidden_layer_sizes=self.hidden_layer_sizes,
                           activation=self.activation,
                           learning_rate=self.learning_rate,
                           learning_rate_init=self.learning_rate_init,
                           momentum=self.momentum,
                           alpha=self.alpha,
                           shuffle=self.shuffle,
                           solver=self.solver)
        self.nn_.fit(X, y)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['nn_'])
        X = check_array(X)
        return self.nn_.predict(X)

    def predict(self, X):
        check_is_fitted(self, ['nn_'])
        X = check_array(X)
        return [1 if pred >= self.threshold else 0
                for pred in self.nn_.predict(X)]

    def score(self, X, y, **kwargs):
        check_X_y(X, y)
        if self.nn_ is None:
            return 0
        y_hat = self.predict(X)
        return accuracy_score(y, y_hat)