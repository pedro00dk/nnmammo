import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_confusion_matrices(matrices, classes, title='Confusion matrix'):
    """
    Plots confusion matrices mean, standard deviation, min and max.

    :param matrices: numpy array with confusion matrices to show.
    :param classes: classes represented in the matrices
    :param title: title of the plot
    """
    plt.clf()

    mean = matrices.mean(axis=0)
    std = matrices.std(axis=0)
    minimum = matrices.min(axis=0)
    maximum = matrices.max(axis=0)

    plt.imshow(mean, interpolation='nearest')
    for i, j in itertools.product(range(mean.shape[0]), range(mean.shape[1])):
        plt.text(j, i, '%.1f±%.1f(%d,%d)' % (mean[i, j], std[i, j], minimum[i, j], maximum[i, j]),
                 horizontalalignment="center")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [str(clazz) for clazz in classes], rotation=45)
    plt.yticks(tick_marks, [str(clazz) for clazz in classes])
    plt.tight_layout()
    plt.colorbar()
    plt.show()


def plot_roc_curves(roc_tuples, title='Receiver Operating characteristic Curve'):
    """
    Plots ROC curves, their mean and standard deviation.

    :param roc_tuples: numpy array with tuples with roc curves data (fpr, tpr, auc).
    :param title: title of the plot
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
    mean_auc = roc_tuples[:, -1].mean()
    std_auc = roc_tuples[:, -1].std()

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.5)
    for i, (fpr, tpr, auc) in enumerate(roc_tuples):
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC rf=%d AUC=%0.2f)' % (i, auc))
    plt.plot(fixed_fpr, mean_tpr, color='b', label=r'Mean ROC AUC=%0.2f±%0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    plt.fill_between(fixed_fpr, std_tpr_lower, std_tpr_upper, color='gray', alpha=.2, label='±1 std. dev.')

    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def plot_samples_configuration_variations_cross_validation_scores_means(samples_scores, xticks, title, xlabel):
    plt.clf()
    x = [*range(len(xticks))]
    plt.xticks(x, xticks)
    for name, configurations_fold_scores in samples_scores.items():
        configurations_mean_fold_scores = [fold_scores.mean() for fold_scores in configurations_fold_scores]
        plt.plot(x, configurations_mean_fold_scores, label=name, marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('fold scores means')
    plt.legend()
    plt.grid()
    plt.show()


def plot_configuration_variation_cross_validation_scores_info(configurations_fold_scores, xticks, title, xlabel):
    plt.clf()
    x = [*range(len(xticks))]
    plt.xticks(x, xticks)
    mins = configurations_fold_scores.min(1)
    maxes = configurations_fold_scores.max(1)
    means = configurations_fold_scores.mean(1)
    std = configurations_fold_scores.std(1)
    plt.plot(x, means)
    plt.errorbar(x, means, std, fmt='ok', lw=8)
    plt.errorbar(x, means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('fold scores means')
    plt.grid()
    plt.show()
