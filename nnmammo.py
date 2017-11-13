# <markdowncell>
# # Analise de configuração da rede MLP da base Mammography
# #### Equipe:
# * João Ricardo dos Santos (jrs4)
# * Pedro Henrique Sousa de Moraes (phsm)

# <markdowncell>
# Leitura da base de dados e separação de instâncias negativas e positivas

# <codecell>
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from extensions import ModMLPClassifier
from plot_util import plot_confusion_matrices, plot_roc_curves
from train_util import validate_model_configurations_variations, sort_configurations_variations_mean_roc_auc

print('reading database')
db_file = 'database.csv'
data_frame = pd.read_csv(db_file, header=None)
print('instances:  %d' % len(data_frame.values))
print()

print('splitting negatives and positives')
negative_instances = data_frame.values[(data_frame.values[:, -1:] == 0).reshape(-1)][:, :-1]
positive_instances = data_frame.values[(data_frame.values[:, -1:] == 1).reshape(-1)][:, :-1]
np.random.shuffle(negative_instances)
np.random.shuffle(positive_instances)
print('negatives: %d, positives: %d' % (len(negative_instances), len(positive_instances)))
print()

# <markdowncell>
# Divisão das instâncias usando k-fold cross validation com 10 folds

# <codecell>
print('creating database folds')
k_folds = 10
negative_instances_folds = np.array_split(negative_instances, k_folds)
positive_instances_folds = np.array_split(positive_instances, k_folds)
folds = [(np.concatenate((negative_instances_folds[i], positive_instances_folds[i])),
          np.concatenate((np.zeros(len(negative_instances_folds[i])), np.ones(len(positive_instances_folds[i])))))
         for i in range(k_folds)]
print('folds: %s' % ['n: %d p: %d' % ((fold[1] == 0).sum(), (fold[1] == 1).sum()) for fold in folds])
print()

# <markdowncell>
# Resample da amostra usando algoritmos de under sampling e over sampling, os algoritmos usados foram
# RandomOverSampler, SMOTE, ClusterCentroids (k-means) e RandomUnderSampler
#
# Os algoritmos são aplicados individualmente em cada fold para que não haja sobreposição dos dados em diferentes folds

# <codecell>
print('creating re-sampled fold copies')
samplers = {
    'r-over': RandomOverSampler(), 'smote': SMOTE(),
    'k-means': ClusterCentroids(), 'r-under': RandomUnderSampler()
}
samples_folds = {name: [sampler.fit_sample(*fold) for fold in folds] for name, sampler in samplers.items()}
for name, sample_folds in samples_folds.items():
    print('sampler %s folds: %s' % (name, [len(fold[0]) for fold in sample_folds]))
print()

# <markdowncell>
# Configurações base, a partir destas, cada atributo será testado independentemente

# <codecell>
model_class = ModMLPClassifier

configuration_constants = {
    'activation': 'logistic',
    'solver': 'sgd',
    'warm_start': False,
    'early_stopping': True,

    # number of training folds (1 will be choose to validate)
    'train_folds': k_folds - 1,
    'max_fail': 2  # max number of consecutive fails in validation score reduction
}

base_configurations = [
    {
        'hidden_layer_sizes': (5,),
        'learning_rate_init': 0.01,
        'learning_rate': 'invscaling',
        'max_iter': 200,
    },
    {
        'hidden_layer_sizes': (10,),
        'learning_rate_init': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 1600,
    },
    {
        'hidden_layer_sizes': (20,),
        'learning_rate_init': 0.0001,
        'learning_rate': 'constant',
        'max_iter': 6400,
    }
]

variations = {
    'hidden_layer_sizes': [(2 ** x,) for x in range(1, 8)],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [200, 400, 800, 1600, 3200, 6400]
}
variations_order = [
    'learning_rate',
    'hidden_layer_sizes',
    'learning_rate_init',
    'max_iter'
]

# activation = ['logistic', 'tanh', 'relu']
# solver = ['lbfgs', 'sgd', 'adam']
# early_stopping = [False, True]

configurations_variations_results = validate_model_configurations_variations(
    model_class,
    samples_folds['k-means'],
    folds,
    [base_configurations[0]],
    'learning_rate',
    variations['learning_rate'],
    runs=1)

plot_confusion_matrices(configurations_variations_results[0][1]['matrix'], ['negative', 'positive'])
plot_roc_curves(configurations_variations_results[0][1]['roc'])

# <codecell>
for name, sample_folds in samples_folds.items():
    print('training %s folds' % name)

    current_configurations = [base_configurations[0]]

    for variation_name in variations_order:
        configurations_variations_results = validate_model_configurations_variations(
            model_class,
            sample_folds,
            folds,
            current_configurations,
            variation_name,
            variations[variation_name],
            runs=1)

        configurations_variations_results = sort_configurations_variations_mean_roc_auc(
            configurations_variations_results)

        current_configurations = [configurations_variations_results[0]]

    print('best configurations')
    print(current_configurations)
    print()
