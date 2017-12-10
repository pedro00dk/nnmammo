import itertools

import numpy as np
from sklearn.metrics import auc, confusion_matrix, mean_squared_error, roc_curve


def validate_model(model, sample_folds, base_folds, test_base_fold=True, runs=1, verbose=1):
    """
    Validates the received model using k-fold cross validation over already split folds and returning a set of
    results (score, mse, matrix, roc).

    :param model: sk-learn model instance to validate
    :param sample_folds: list of tuples with two lists each, one of instances and other of instance's classes
    :param base_folds: same as sample folds but with no processing, can be None if test_original_folds is False
    :param test_base_fold: if should use sample_folds or original_original_folds test set
    :param runs: number of runs over all folds
    :param verbose: increases output verbosity
    :return: a dictionary with result types as keys and the result as values in arrays of runs * k_folds size
    """
    results = {'score': [], 'mse': [], 'matrix': [], 'roc': {'fpr': [], 'tpr': [], 'auc': []}}

    for run, i in itertools.product(range(runs), range(len(sample_folds))):
        if verbose > 2:
            print(f'run {run + 1} of {runs}, fold {i + 1} of {len(sample_folds)}')

        train_folds = sample_folds[i + 1:len(sample_folds)] + sample_folds[0:i]
        train_instances = [instance for fold in train_folds for instance in fold[0]]
        train_classes = [clazz for fold in train_folds for clazz in fold[1]]

        test_instances = base_folds[i][0] if test_base_fold else sample_folds[i][0]
        test_classes = base_folds[i][1] if test_base_fold else sample_folds[i][1]

        model.fit(train_instances, train_classes)
        results['score'].append(model.score(test_instances, test_classes))

        probabilities = model.predict_proba(test_instances)
        results['mse'].append(mean_squared_error(test_classes, np.round(probabilities[:, -1])))
        results['matrix'].append(confusion_matrix(test_classes, np.round(probabilities[:, -1])))
        fpr, tpr, _ = roc_curve(test_classes, probabilities[:, -1])
        area = auc(fpr, tpr)
        results['roc']['fpr'].append(fpr)
        results['roc']['tpr'].append(tpr)
        results['roc']['auc'].append(area)

    return results


def validate_model_configurations_variations(model_class, configurations, variation_name, variation_values,
                                             sample_folds, base_folds, test_base_fold=True, runs=1,
                                             verbose=1):
    """
    Tests multiples variations of multiples configurations of a model using k-fold cross validation over already split
    folds returning information about the variations.

    :param model_class: sk-learn model class to validate
    :param configurations: base of the configurations to test the variations
    :param variation_name: variation attribute name in model configurations
    :param variation_values: variation values to be tested
    :param sample_folds: list of tuples with two lists each, one of instances and other of instance's classes
    :param base_folds: same as sample folds but with no processing, can be None if test_original_folds is False
    :param test_base_fold: if should use sample_folds or original_original_folds test set
    :param runs: number of runs over all folds
    :param verbose: increases output verbosity
    :return: list of tuples with the variation and dictionary with score, mse, matrix and roc data of test results.
    """
    configurations_results = []

    for i, configuration in enumerate(configurations):
        if verbose > 0:
            print(f'configuration {i + 1} of {len(configurations)}')
            print(f'-> {configuration}')

        for j, variation_value in enumerate(variation_values):
            if verbose > 1:
                print(f'variation {j + 1} of {len(variation_values)} | {variation_name} -> {variation_value}')

            configuration = configuration.copy()
            configuration[variation_name] = variation_value
            model = model_class(**configuration)
            result = validate_model(model, sample_folds, base_folds, test_base_fold, runs, verbose)
            configurations_results.append((configuration, result))

    return configurations_results
