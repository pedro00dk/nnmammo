import itertools

import numpy as np
from sklearn.metrics import auc, confusion_matrix, mean_squared_error, roc_curve


def validate_model(model, sample_folds, original_folds, runs=5, test_original_fold=True,
                   result_data={'score', 'preds', 'probs', 'mse', 'matrix', 'roc'}, verbose=True):
    """
    Validates the received model using k-fold cross validation over already split folds and returning a set of
    selectable results.

    :param model: scikit-learn model instance to validate
    :param sample_folds: list of tuples with two lists each, one of instances and other of instance's classes
    :param original_folds: same as sample folds but with no processing, can be None if test_original_folds is False
    :param runs: number of runs over all folds
    :param test_original_fold: if should use sample_folds or original_original_folds test set
    :param result_data: set with expected results. Options: 'score', 'preds', 'probs', 'mse', 'matrix' and 'roc'
    :param verbose: if should print the current iteration
    :return: a dictionary with expected_results as keys and the result as values in arrays of runs * k_folds size
    """
    results = {name: [] for name in result_data}
    for run, i in itertools.product(range(runs), range(len(sample_folds))):
        if verbose:
            print('run %d of %d, fold %d of %d' % (run + 1, runs, i + 1, len(sample_folds)))
        train_folds = sample_folds[i + 1:len(sample_folds)] + sample_folds[0:i]
        train_instances = [instance for fold in train_folds for instance in fold[0]]
        train_classes = [clazz for fold in train_folds for clazz in fold[1]]
        test_instances = original_folds[i][0] if test_original_fold else sample_folds[i][0]
        test_classes = original_folds[i][1] if test_original_fold else sample_folds[i][1]
        model.fit(train_instances, train_classes)

        if 'score' in results:
            results['score'].append(model.score(test_instances, test_classes))
        if 'preds' in results:
            results['preds'].append(model.predict(test_instances))
        if 'probs' in results:
            results['probs'].append(model.predict_proba(test_instances))

        if 'mse' in results or 'matrix' in results or 'roc' in results:
            probs = results['probs'][-1] if 'probs' in results else model.predict_proba(test_instances)

        if 'mse' in results:
            results['mse'].append(mean_squared_error(test_classes, np.round(probs[:, -1])))
        if 'matrix' in results:
            results['matrix'].append(confusion_matrix(test_classes, np.round(probs[:, -1])))
        if 'roc' in results:
            fpr, tpr, _ = roc_curve(test_classes, probs[:, -1])
            area = auc(fpr, tpr)
            results['roc'].append([fpr, tpr, area])

    results = {name: np.array(r) if name != 'roc' else [np.array(d) for d in r] for name, r in results.items()}
    return results


def validate_model_configuration_variations(model_class, sample_folds, original_folds, base_configuration,
                                            variation_name, variation_values, runs=5, test_original_fold=True,
                                            verbose=True):
    """
    Tests multiples variations of a configuration of a model using k-fold cross validation over already split folds
    returning information about the variations.

    :param model_class: scikit-learn model class to validate
    :param sample_folds: list of tuples with two lists each, one of instances and other of instance's classes
    :param original_folds: same as sample folds but with no processing, can be None if test_original_folds is False
    :param base_configuration: base of the configuration to test
    :param variation_name: variation attribute name in model
    :param variation_values: variation values to be tested
    :param runs: number of runs over all folds
    :param test_original_fold: if should use sample_folds or original_original_folds test set
    :param verbose: if should print the current iteration
    :return: list of tuples with the configuration and dictionary with mse, matrix and roc keys and corresponding arrays
             of test results.
    """
    configuration_variations_results = []
    result_data = {'mse', 'matrix', 'roc'}
    for i, variation_value in enumerate(variation_values):
        if verbose:
            print('variation %d of %d | %s -> %s' % (i + 1, len(variation_values), variation_name, variation_value))
        configuration = base_configuration.copy()
        configuration[variation_name] = variation_value
        model = model_class(**configuration)
        result = validate_model(model, sample_folds, original_folds, runs, test_original_fold, result_data, verbose)
        configuration_variations_results.append((configuration, result))
    return configuration_variations_results


def validate_model_configurations_variations(model_class, sample_folds, original_folds, configurations, variation_name,
                                             variation_values, runs=5, test_original_fold=True, verbose=True):
    """
    Tests multiples variations of multiples configurations of a model using k-fold cross validation over already split
    folds returning information about the variations.

    :param model_class: scikit-learn model class to validate
    :param sample_folds: list of tuples with two lists each, one of instances and other of instance's classes
    :param original_folds: same as sample folds but with no processing, can be None if test_original_folds is False
    :param configurations: configurations to test the variations
    :param variation_name: variation attribute name in model
    :param variation_values: variation values to be tested
    :param runs: number of runs over all folds
    :param test_original_fold: if should use sample_folds or original_original_folds test set
    :param verbose: if should print the current iteration
    :return: list of tuples with the configuration and dictionary with mse, matrix and roc keys and corresponding arrays
             of test results.
    """
    configurations_variations_results = []
    for i, configuration in enumerate(configurations):
        if verbose:
            print('configuration %d of %d' % (i + 1, len(configurations)))
            print('-> %s' % configuration)
        result = validate_model_configuration_variations(model_class, sample_folds, original_folds, configuration,
                                                         variation_name,
                                                         variation_values, runs, test_original_fold, verbose)
        configurations_variations_results.extend(result)
    return configurations_variations_results
