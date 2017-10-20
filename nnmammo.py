# <markdowncell>
# # Analise de configuração da rede MLP da base Mammography
# #### Equipe:
# * João Ricardo dos Santos (jrs4)
# * Pedro Henrique Sousa de Moraes (phsm)

# <markdowncell>
# Leitura da base de dados

# <codecell>
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from extended_support import cross_validate_model_configuration_variations_scores, \
    plot_configuration_variation_cross_validation_scores_info, \
    plot_samples_configuration_variations_cross_validation_scores_means, ModMLPClassifier

print('reading database')

db_file = 'database.csv'

df = pd.read_csv(db_file, header=None)

# <codecell>
print('spliting database in folds')

k_folds = 10

negative_instances = df.values[(df.values[:, -1:] == 0).reshape(-1)][:, :-1]
positive_instances = df.values[(df.values[:, -1:] == 1).reshape(-1)][:, :-1]
np.random.shuffle(negative_instances)
np.random.shuffle(positive_instances)
print('negatives: %d, positives: %d' % (len(positive_instances), len(negative_instances)))

folds_negative_instances = np.array_split(negative_instances, k_folds)
folds_positive_instances = np.array_split(positive_instances, k_folds)

folds = [(np.concatenate((folds_negative_instances[i], folds_positive_instances[i])),
          np.concatenate((np.zeros(len(folds_negative_instances[i])), np.ones(len(folds_positive_instances[i])))))
         for i in range(k_folds)]

for index, (instances, classes) in enumerate(folds):
    print('\t fold %d contains %d instances, %d positives, %d negatives' %
          (index, len(instances), (classes == 0).sum(), (classes == 1).sum()))

# <markdowncell>
# Resampling da amostra usando algoritmos de under sampling e over sampling, os algoritmos usados foram
# RandomOverSampler, SMOTE, ClusterCentroids (k-means) e RandomUnderSampler

# <codecell>
print('creating re-sampled fold copies')
samplers = {
    'rover': RandomOverSampler(), 'smote': SMOTE(),
    'kmeans': ClusterCentroids(), 'runder': RandomUnderSampler()
}
samples_folds = {name: [sampler.fit_sample(*fold) for fold in folds] for name, sampler in samplers.items()}
for name, folds in samples_folds.items():
    print('sampler %s fold sizes %s' % (name, [len(fold[0]) for fold in folds]))

# <markdowncell>
# Default configuration and variations for tests

# <codecell>
configuration = {
    'hidden_layer_sizes': (8,),
    'activation': 'logistic',
    'solver': 'sgd',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_iter': 1600,
    'early_stopping': True,
    # non variable configurations
    'train_folds': k_folds - 1,  # number of training folds (1 will be choose to validate)
    'warm_start': False  # re-initializes the network in consecutive fit calls

}
hidden_layer_sizes = [(x,) for x in range(1, 21)]
activation = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']
learning_rate_init = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
max_iter = [200, 400, 800, 1600, 3200, 6400, ]
early_stopping = [False, True]

model_class = ModMLPClassifier

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
samples_scores = {}
for (name, sample_folds) in samples_folds.items():
    print('sampler %s:' % name)
    samples_scores[name] = cross_validate_model_configuration_variations_scores(sample_folds, folds, model_class,
                                                                                configuration, 'hidden_layer_sizes',
                                                                                hidden_layer_sizes, verbose=True)

plot_samples_configuration_variations_cross_validation_scores_means(samples_scores, hidden_layer_sizes,
                                                                    'hidden layer sizes scores', 'sizes')
for name, configurations_fold_scores in samples_scores.items():
    plot_configuration_variation_cross_validation_scores_info(configurations_fold_scores, hidden_layer_sizes,
                                                              '%s specific hidden layer sizes results' % name,
                                                              'sizes')

# <codecell>
samples_scores = {}
requirements = {'solver': 'sgd', 'power_t': 1}
for (name, sample_folds) in samples_folds.items():
    print('sampler %s:' % name)
    samples_scores[name] = cross_validate_model_configuration_variations_scores(sample_folds, folds, model_class,
                                                                                configuration, 'learning_rate',
                                                                                learning_rate, requirements,
                                                                                verbose=True)

plot_samples_configuration_variations_cross_validation_scores_means(samples_scores, learning_rate,
                                                                    'learning rate rules scores', 'rules')
for name, configurations_fold_scores in samples_scores.items():
    plot_configuration_variation_cross_validation_scores_info(configurations_fold_scores, learning_rate,
                                                              '%s specific learning rate rules results' % name,
                                                              'rules')
