import itertools

import matplotlib.pyplot as plt
import numpy as np


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


def plot_roc_curve(roc_tuples, classes, title='Receiver Operating characteristic Curve'):
    # Not working yet

    plt.clf()

    mean = roc_tuples.mean(axis=(0, 1))
    std = roc_tuples.std(axis=(0, 1))

    for i, (fpr, tpr, thresholds, auc) in enumerate(roc_tuples):
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC  rf=%d AUC=%0.2f)' % (i, auc))

    plt.plot(mean[0], mean[1], color='b', label='Mean ROC AUC=%0.2f±%0.2f)' % (mean[3], std[3]), lw=2, alpha=0.8)

    mean_fpr = np.linspace(0, 1, 100)
    tprs_std_upper = np.minimum(mean[1] + std[1], 1)
    tprs_std_lower = np.maximum(mean[1] - std[1], 0)

    plt.fill_between(mean_fpr, tprs_std_upper, tprs_std_lower, color='gray', alpha=.2, label='±1 std. dev.')

    plt.legend(loc="lower right")
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
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
