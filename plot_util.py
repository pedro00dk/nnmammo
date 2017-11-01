import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_confusion_matrices(matrices, classes, title='Confusion matrix'):
    """
    Single configuration plot.
    Plots k-fold cross validation confusion matrices mean, standard deviation, min and max.

    :param matrices: numpy array with confusion matrices to show.
    :param classes: classes represented in the matrices
    :param title: plot title
    """
    plt.clf()

    tick_marks = np.arange(len(classes))

    mean = np.mean(matrices, axis=0)
    std = np.std(np.asarray(matrices, float), axis=0)
    minimum = np.min(matrices, axis=0)
    maximum = np.max(matrices, axis=0)

    plt.imshow(mean, interpolation='nearest')
    for i, j in itertools.product(range(mean.shape[0]), range(mean.shape[1])):
        plt.text(j, i, '%.1f±%.1f(%d,%d)' % (mean[i, j], std[i, j], minimum[i, j], maximum[i, j]),
                 horizontalalignment="center")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(tick_marks, [str(clazz) for clazz in classes], rotation=45)
    plt.yticks(tick_marks, [str(clazz) for clazz in classes])
    plt.tight_layout()
    plt.colorbar()
    plt.show()


def plot_roc_curves(roc_tuples, title='Receiver Operating characteristic Curve'):
    """
    Single configuration plot.
    Plots k-fold cross validation ROC curves, their mean and standard deviation.

    :param roc_tuples: numpy array with tuples with roc curves data (fpr, tpr, auc).
    :param title: plot title
    """
    plt.clf()

    fixed_fpr = np.linspace(0, 1, 200)
    fixed_tprs = [sp.interp(fixed_fpr, fpr, tpr) for fpr, tpr, auc in roc_tuples]
    for fixed_tpr in fixed_tprs:
        fixed_tpr[0] = 0
    mean_tpr = np.mean(fixed_tprs, axis=0)
    mean_tpr[-1] = 1
    std_tpr = np.std(fixed_tprs, axis=0)
    std_tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    std_tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    aucs = [auc for fpr, tpr, auc in roc_tuples]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Luck', alpha=0.5)
    for i, (fpr, tpr, auc) in enumerate(roc_tuples):
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC rf=%d AUC=%0.2f)' % (i, auc))
    plt.plot(fixed_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label='Mean ROC AUC=%0.2f±%0.2f)' % (mean_auc, std_auc))
    plt.fill_between(fixed_fpr, std_tpr_lower, std_tpr_upper, color='gray', alpha=0.2, label='±1 std. dev.')

    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def plot_configurations_variations_single_value(values, variation_name, variation_values, title, ylabel):
    """
    Plot a value variation in multiple configurations (bi-dimensional array).

    :param values: values to plot (array of arrays)
    :param variation_name: the varying property name
    :param variation_values: the varying value that generated aech sub array in values
    :param title: plot title
    :param ylabel: y axis label (value interpretation)
    """
    plt.clf()

    tick_marks = np.arange(len(variation_values))

    means = np.mean(values, axis=1)
    stds = np.std(values, axis=1)
    minimums = np.min(values, axis=1)
    maximums = np.max(values, axis=1)

    plt.errorbar(tick_marks, means, stds, fmt='ok', lw=6, alpha=0.8)
    plt.errorbar(tick_marks, means, [means - minimums, maximums - means], fmt='.k', ecolor='gray', lw=3, alpha=0.8)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Configuration Variations (%s)' % variation_name)
    plt.xticks(tick_marks, [str(value) for value in variation_values])
    plt.grid()
    plt.show()
