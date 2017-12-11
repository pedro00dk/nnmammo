# <markdowncell>
# # Analise de configuração da rede MLP com a base Mammography

# <markdowncell>
# Leitura da base de dados e separação de instâncias negativas e positivas

# <codecell>
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from extensions import *
from plot_util import *
from train_util import *

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
base_folds = [(np.concatenate((negative_instances_folds[i], positive_instances_folds[i])),
               np.concatenate((np.zeros(len(negative_instances_folds[i])), np.ones(len(positive_instances_folds[i])))))
              for i in range(k_folds)]
print('folds: %s' % ['n: %d p: %d' % ((fold[1] == 0).sum(), (fold[1] == 1).sum()) for fold in base_folds])
print()

# <markdowncell>
# Resample da amostra usando algoritmos de under sampling e over sampling, os algoritmos usados foram
# RandomOverSampler, SMOTE, ClusterCentroids (K-Means) e RandomUnderSampler
#
# Os algoritmos são aplicados individualmente em cada fold para que não haja sobreposição dos dados em diferentes folds

# <codecell>
print('creating re-sampled fold copies')
samplers = {
    'r-over': RandomOverSampler(), 'smote': SMOTE(),
    'k-means': ClusterCentroids(), 'r-under': RandomUnderSampler()
}
samples_folds = {name: [sampler.fit_sample(*fold) for fold in base_folds] for name, sampler in samplers.items()}
for name, sample_folds in samples_folds.items():
    print('sampler %s folds: %s' % (name, [len(fold[0]) for fold in sample_folds]))
print()

# <markdowncell>
# Configurações base, a partir destas, cada atributo será testado independentemente

# <codecell>
model_class = ModMLPClassifier

base_configurations = [
    {
        'solver': 'sgd',
        'activation': 'logistic',
        'early_stopping': True,
        'hidden_layer_sizes': (5,),
        'learning_rate_init': 0.01,
        'learning_rate': 'invscaling',
        'max_iter': 200,
        'warm_start': False,  # reset the model when fit is call
        'train_folds': k_folds - 1,  # number of training folds (1 will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    },
    {
        'solver': 'sgd',
        'activation': 'logistic',
        'early_stopping': True,
        'hidden_layer_sizes': (10,),
        'learning_rate_init': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 800,
        'warm_start': False,  # reset the model when fit is call
        'train_folds': k_folds - 1,  # number of training folds (1 will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    },
    {
        'solver': 'sgd',
        'activation': 'logistic',
        'early_stopping': True,
        'hidden_layer_sizes': (20,),
        'learning_rate_init': 0.0001,
        'learning_rate': 'constant',
        'max_iter': 3200,
        'warm_start': False,  # reset the model when fit is call
        'train_folds': k_folds - 1,  # number of training folds (1 will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    }
]

variations = [
    ('learning_rate_init', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
    ('hidden_layer_sizes', [(2 ** x,) for x in range(1, 8)]),
    ('max_iter', [200, 400, 800, 1600, 3200, 6400]),
    ('learning_rate', ['constant', 'invscaling', 'adaptive'])
]

print('base configurations:')
print(pd.DataFrame(base_configurations).to_string())
print()

# <markdowncell>
# Testes individuais com as diferentes bases geradas com os algoritmos de sampling
# * K-Means

configuration_range = 3
print(f'optimize model configuration for k-means with configuration range of {configuration_range}')
configurations = base_configurations
results = None
for i, (variation_name, variation_values) in enumerate(variations):
    print(f'variation {i} -> {variation_name}')
    configurations_results = validate_model_configurations_variations(model_class, configurations,
                                                                      variation_name, variation_values,
                                                                      samples_folds['k-means'], base_folds,
                                                                      verbose=1)

    data_frame = pd.DataFrame([configuration for configuration, result in configurations_results])
    data_frame['mean score'] = [np.mean(result['score']) for _, result in configurations_results]
    data_frame['mean mse'] = [np.mean(result['mse']) for _, result in configurations_results]
    data_frame['mean roc auc'] = [np.mean(result['roc']['auc']) for _, result in configurations_results]
    data_frame['data'] = [(c, r) for c, r in configurations_results]

    data_frame.sort_values(by=['mean roc auc', 'mean score', 'mean mse'], ascending=[False, False, True], inplace=True)
    filtered_data_frame = data_frame[[column for column in data_frame if column != 'data']]
    configurations = [c for c, _ in data_frame['data'][:min(configuration_range, len(data_frame['data']))]]
    results = [r for _, r in data_frame['data'][:min(configuration_range, len(data_frame['data']))]]
    print(filtered_data_frame.to_string())
    plot_multi_configuration_roc_curves((r['roc'] for r in results), data_frame.index)
    print()
    break

best_configuration = configurations[0]
best_configuration_results = results[0]
plot_single_configuration_roc_curves(best_configuration_results['roc'],
                                     title='Best configuration for k-means ROC Curve')
plot_single_configuration_confusion_matrix(best_configuration_results['matrix'], ['neg', 'pos'],
                                           title='Best configuration for k-means Confusion Matrix')
