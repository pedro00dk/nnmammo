# -*- coding: utf-8 -*-
# <nbformat>4.0</nbformat>

# <markdowncell>
# # Analise de configuração da rede MLP da base Mammography
# #### Equipe:
# * João Ricardo dos Santos (jrs4)
# * Pedro Henrique Sousa de Moraes (phsm)

# <markdowncell>
# Leitura da base de dados

# <codecell>
import numpy as np

db = ([], [])  # (instances, classes)
with open('mammo.csv') as db_file:
    for line in db_file:
        instance = np.array([float(attribute)
                             for attribute in line.split(',')])
        db[0].append(instance[:len(instance) - 1])
        db[1].append(instance[len(instance) - 1])

print('negatives: %d, positives: %d' % (db[1].count(0), db[1].count(1)))

# <markdowncell>
# Resampling da amostra usando algoritmos de under sampling e over sampling, os algoritmos usados foram RandomOverSampler, SMOTE, AllKNN e RandomUnderSampler

# <codecell>
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import AllKNN, RandomUnderSampler

samplers = {
    'rover': RandomOverSampler(), 'smote': SMOTE(),
    'aknn': AllKNN(n_neighbors=180, n_jobs=-1), 'runder': RandomUnderSampler()
}
db_samples = {name: sampler.fit_sample(*db)
              for name, sampler in samplers.items()}
for name, db_sample in db_samples.items():
    print('%s: negatives: %d, positives: %d' %
          (name, (db_sample[1] == 0).sum(), (db_sample[1] == 1).sum()))

# <codecell>
configuration = {
    'hidden_layer_sizes': (8,),
    'activation': 'logistic',
    'solver': 'sgd',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_iter': 1600,
    'early_stopping': True,
    'validation_fraction': 1 / 9
}
hidden_layer_sizes = [(x,) for x in range(1, 21)]
activation = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_init = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
max_iter = [200, 400, 800, 1600, 3200, 6400, ]
early_stopping = [False, True]

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
# * Método de agrupamento para redução do conjunto

# <codecell>
import itertools

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

skf = StratifiedKFold(n_splits=10, shuffle=True)
hidden_layer_sizes_means = {}
for (name, db_sample), size in itertools.product(db_samples.items(), hidden_layer_sizes):
    configuration_copy = configuration.copy()
    configuration_copy['hidden_layer_sizes'] = size
    model = MLPClassifier(**configuration_copy)
    score = cross_val_score(model, *db_sample, cv=skf, verbose=False)
    if name not in hidden_layer_sizes_means:
        hidden_layer_sizes_means[name] = []
    hidden_layer_sizes_means[name].append(score.mean())
    print('%s %s: %s' % (name, size, score))

# <codecell>
import matplotlib.pyplot as plt

plt.clf()
x = [*range(len(hidden_layer_sizes))]
plt.xticks(x, hidden_layer_sizes)
for name, means in hidden_layer_sizes_means.items():
    plt.plot(x, means, label=name, marker='x')
plt.title('hidden layer sizes scores per sampling configuration')
plt.xlabel('sizes')
plt.ylabel('score')
plt.legend()
plt.grid()
plt.show()

# <codecell>
learning_rate_means = {}
for (name, db_sample), learning_rate_rule in itertools.product(db_samples.items(), learning_rate):
    configuration_copy = configuration.copy()
    
    # required for learning rate rules
    configuration_copy['solver']: 'sgd'
    configuration_copy['power_t']: 1

    configuration_copy['learning_rate'] = learning_rate_rule
    model = MLPClassifier(**configuration_copy)
    score = cross_val_score(model, *db_sample, cv=skf, verbose=False)
    if name not in learning_rate_means:
        learning_rate_means[name] = []
    learning_rate_means[name].append(score.mean())
    print('%s %s: %s' % (name, learning_rate_rule, score))

# <codecell>
plt.clf()
x = [*range(len(learning_rate))]
plt.xticks(x, learning_rate)
for name, means in learning_rate_means.items():
    plt.plot(x, means, label=name, marker='x')
plt.title('learning rate rules scores per sampling configuration')
plt.xlabel('rules')
plt.ylabel('score')
plt.legend()
plt.grid()
plt.show()
