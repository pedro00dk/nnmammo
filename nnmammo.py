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
from train_util import validate_model_configurations

print('reading database')
DB_FILE = 'database.csv'
DATA_FRAME = pd.read_csv(DB_FILE, header=None)
print('instances:  %d' % len(DATA_FRAME.values))
print()

print('splitting negatives and positives')
negative_instances = DATA_FRAME.values[(DATA_FRAME.values[:, -1:] == 0).reshape(-1)][:, :-1]
positive_instances = DATA_FRAME.values[(DATA_FRAME.values[:, -1:] == 1).reshape(-1)][:, :-1]
np.random.shuffle(negative_instances)
np.random.shuffle(positive_instances)
print('negatives: %d, positives: %d' % (len(negative_instances), len(positive_instances)))
print()

# <markdowncell>
# Divisão das instâncias usando k-fold cross validation com 10 folds

# <codecell>
print('creating database folds')
k_folds = 10
folds_negative_instances = np.array_split(negative_instances, k_folds)
folds_positive_instances = np.array_split(positive_instances, k_folds)
folds = [(np.concatenate((folds_negative_instances[i], folds_positive_instances[i])),
          np.concatenate((np.zeros(len(folds_negative_instances[i])), np.ones(len(folds_positive_instances[i])))))
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

    'train_folds': k_folds - 1,  # number of training folds (1 will be choose to validate)
    'max_fail': 5  # max number of consecutive fails in validation score reduction
}

base_configurations = [
    {
        'hidden_layer_sizes': (5,),
        'learning_rate_init': 0.05,
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

test_attributes = {
    'hidden_layer_sizes': [(x,) for x in range(1, 21)],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [200, 400, 800, 1600, 3200, 6400]
}
test_attributes_order = ['hidden_layer_sizes', 'learning_rate_init', 'learning_rate', 'max_iter']

# <codecell>
variation_results = validate_model_configurations(model_class, samples_folds['smote'], folds, base_configurations[1],
                                                  'learning_rate', [test_attributes['learning_rate'][0]], runs=1,
                                                  test_original_fold=True)

from plot_util import plot_confusion_matrices, plot_roc_curve


plot_confusion_matrices(variation_results[0]['matrix'], ['negatives', 'positives'])

import matplotlib.pyplot as plt
plot_roc_curve(variation_results[0]['roc'], ['negatives', 'positives'])



# activation = ['logistic', 'tanh', 'relu']
# solver = ['lbfgs', 'sgd', 'adam']
# early_stopping = [False, True]


# <codecell>


# <markdowncell>
# #### Investigar diferentes topologias da rede e diferentes valores de parâmetros (básico)
# * Tamanho do conjunto de dados
# * Número de unidades intermediárias
# * Influência da taxa de aprendizagem no treinamento
# * Overfitting (memorização do conjunto de treinamento)
#
# #### Investigar parâmetros adicionais
# * Algoritmo
# * Função de ativação
# * Taxa adaptativa
# * Método de agrupamento para redução