import warnings

import numpy as np
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
# Internal needed packages
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.utils import gen_batches, shuffle


class ModMLPClassifier(MLPClassifier):
    """
    Extension of MLPClassifier class in scikit-learn.

    This extension supports the new  parameters train_folds and max_fail.

    param train_folds: number of folds merged in training set.

    If train_folds is None the default train_test_split is used to obtain the validation set based on validation
    fraction parameter, if is a number that indicates the number of folds of the training set: the last fold is selected
    as validation fold.

    param max_fail: overrides the default constant value (2) of max fail in the scikit-learn implementation
    """

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 train_folds=None, max_fail=5):
        super().__init__(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
                         power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum,
                         nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon)

        # EXTENSION added properties
        self.train_folds = train_folds
        self.max_fail = max_fail

    def _fit_stochastic(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units, incremental):
        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental
        if early_stopping:

            # EXTENSION train_folds (modifications here)
            X, X_val, y, y_val = self._split_train_validation(X, y)

            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                X, y = shuffle(X, y, random_state=self._random_state)
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    activations[0] = X[batch_slice]
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                # EXTENSION max_fail (modified  next line)
                if self._no_improvement_count > self.max_fail:
                    # not better than last two iterations by tol.
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Validation score did not improve more than "
                               "tol=%f for two consecutive epochs." % self.tol)
                    else:
                        msg = ("Training loss did not improve more than tol=%f"
                               " for two consecutive epochs." % self.tol)

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    # EXTENSION train_folds (new function)
    def _split_train_validation(self, X, y):
        if self.train_folds is None or self.train_folds == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=self._random_state,
                test_size=self.validation_fraction)
        else:
            split_index = int(len(y) * ((self.train_folds - 1) / self.train_folds))
            X_train = X[0:split_index]
            X_test = X[split_index: X.shape[0]]
            y_train = y[0:split_index]
            y_test = y[split_index: X.shape[0]]
        return X_train, X_test, y_train, y_test
