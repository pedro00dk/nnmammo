import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_single_configuration_roc_curves(roc_dict, title='Receiver Operating Characteristic Curve'):
    """
    Single result data.

    Plots the ROC curve information of a single configuration, roc_dicts should contains three elements
    ('fpr', 'tpr', 'auc'), these elements should be lists with the associated information of multiple executions.

    :param roc_dict: roc curve data
    :param title: plot title
    """
    plt.clf()

    fixed_fpr = np.linspace(0, 1, 500)
    fixed_tprs = [sp.interp(fixed_fpr, fpr, tpr) for fpr, tpr in zip(roc_dict['fpr'], roc_dict['tpr'])]
    for fixed_tpr in fixed_tprs:
        fixed_tpr[0] = 0
    mean_tpr = np.mean(fixed_tprs, axis=0)
    mean_tpr[-1] = 1
    std_tpr = np.std(fixed_tprs, axis=0)
    std_tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    std_tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    mean_auc = np.mean(roc_dict['auc'])
    std_auc = np.std(roc_dict['auc'])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Luck', alpha=0.5)
    for i in range(len(roc_dict['auc'])):
        fpr = roc_dict['fpr'][i]
        tpr = roc_dict['tpr'][i]
        auc = roc_dict['auc'][i]
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fr={i} AUC={auc:.2f}')
    plt.plot(fixed_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC AUC={mean_auc:.2f}±{std_auc:.2f}')
    plt.fill_between(fixed_fpr, std_tpr_lower, std_tpr_upper, color='gray', alpha=0.2, label='±1 std. dev.')

    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def plot_multi_configuration_roc_curves(roc_dicts, labels, title='Receiver Operating Characteristic Curves'):
    """
    Multi result data.

    Plots the ROC curve information of multiple configurations (mean), each element in roc_dicts should contains three
    elements ('fpr', 'tpr', 'auc'), these elements should be lists with the associated information of multiple
    executions.

    :param roc_dicts: roc curves data
    :param labels: labels for each curve
    :param title: plot title
    """
    plt.clf()

    fixed_fpr = np.linspace(0, 1, 500)
    for i, roc_dict in enumerate(roc_dicts):
        fixed_tprs = [sp.interp(fixed_fpr, fpr, tpr) for fpr, tpr in zip(roc_dict['fpr'], roc_dict['tpr'])]
        for fixed_tpr in fixed_tprs:
            fixed_tpr[0] = 0
        mean_tpr = np.mean(fixed_tprs, axis=0)
        mean_tpr[-1] = 1
        mean_auc = np.mean(roc_dict['auc'])
        std_auc = np.std(roc_dict['auc'])
        plt.plot(fixed_fpr, mean_tpr, lw=2, alpha=0.8, label=f'{labels[i]} AUC={mean_auc:.2f}±{std_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Luck', alpha=0.5)

    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def plot_single_configuration_confusion_matrix(matrices, classes, normalize=True, title='Confusion matrix'):
    """
    Single configuration plot.
    Plots k-fold cross validation confusion matrices mean, standard deviation, min and max.

    :param matrices: numpy array with confusion matrices to show.
    :param classes: classes represented in the matrices
    :param normalize: is the plot should be normalized
    :param title: plot title
    """
    plt.clf()

    if normalize:
        matrices = [m.astype('float') / m.sum(axis=1)[:, np.newaxis] for m in matrices]

    mean = np.mean(matrices, axis=0)
    std = np.std(np.asarray(matrices, float), axis=0)

    plt.imshow(mean, interpolation='nearest')
    for i, j in itertools.product(range(mean.shape[0]), range(mean.shape[1])):
        text = f'{mean[i, j]:.3f}±{std[i, j]:.3f}' if normalize else f'{mean[i, j]:d}±{std[i, j]:d}'
        plt.text(j, i, text, horizontalalignment="center")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [str(clazz) for clazz in classes], rotation=45)
    plt.yticks(tick_marks, [str(clazz) for clazz in classes])
    plt.tight_layout()
    plt.colorbar()
    plt.show()
