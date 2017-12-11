# <markdowncell>
# # Analise de configuração da rede MLP com a base Mammography

# <markdowncell>
# Leitura da base de dados

# <codecell>
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from extensions import *
from IPython.display import display
from plot_util import *
from train_util import *

NOTEBOOK = False

print('read database')
db_file = 'database.csv'
db_frame = pd.read_csv(db_file, header=None)
print(f'instances: {db_frame.shape[0]}')
print()

# <markdowncell>
# Separação de instâncias positivas e negativas


# <codecell>
print('split negatives and positives')
negs_frame = db_frame[db_frame.iloc[:, -1] == 0]
poss_frame = db_frame[db_frame.iloc[:, -1] == 1]
print('shuffle negatives and positives')
negs_frame = negs_frame.sample(frac=1).reset_index(drop=True)
poss_frame = poss_frame.sample(frac=1).reset_index(drop=True)
print(f'neg: {negs_frame.shape[0]}, pos: {poss_frame.shape[0]}')
print()

# <markdowncell>
# Divisão das instâncias usando k-fold cross validation com 10 folds

# <codecell>
k = 10
print(f'create k={k} folds')
negs_frame_folds = np.array_split(negs_frame, k)
poss_frame_folds = np.array_split(poss_frame, k)
print('merge and shuffle individual folds')
frame_folds = [pd.concat([negs_frame_folds[i], poss_frame_folds[i]], axis=0, join='outer') for i in range(k)]
frame_folds = [ff.sample(frac=1).reset_index(drop=True) for ff in frame_folds]
print(f'split instances and classes and transform in numpy arrays')
base_folds = [(ff.iloc[:, :-1].as_matrix(), ff.iloc[:, -1].as_matrix()) for ff in frame_folds]
print('\n'.join(f'{i}: n={negs_frame_folds[i].shape[0]} p={poss_frame_folds[i].shape[0]} t={frame_folds[i].shape[0]}'
                for i in range(k)))
print()

# <markdowncell>
# Resample da amostra usando algoritmos de under sampling e over sampling, os algoritmos usados foram
# RandomOverSampler, SMOTE, ClusterCentroids (K-Means) e RandomUnderSampler
#
# Os algoritmos são aplicados individualmente em cada fold para que não haja sobreposição dos dados em diferentes folds

# <codecell>
print('resample folds')
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
        'solver': 'sgd', 'activation': 'logistic', 'early_stopping': True, 'hidden_layer_sizes': (8,),
        'learning_rate_init': 0.01, 'learning_rate': 'invscaling', 'max_iter': 200,

        'train_folds': k - 1,  # number of training folds (the last will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    },
    {
        'solver': 'sgd', 'activation': 'logistic', 'early_stopping': True, 'hidden_layer_sizes': (16,),
        'learning_rate_init': 0.001, 'learning_rate': 'adaptive', 'max_iter': 800,

        'train_folds': k - 1,  # number of training folds (the last will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    },
    {
        'solver': 'sgd', 'activation': 'logistic', 'early_stopping': True, 'hidden_layer_sizes': (32,),
        'learning_rate_init': 0.0001, 'learning_rate': 'constant', 'max_iter': 3200,

        'train_folds': k - 1,  # number of training folds (the last will be choose to validate)
        'max_fail': 3  # max number of consecutive fails in validation score reduction
    }
]
print('base configurations:')
if NOTEBOOK:
    display(pd.DataFrame(base_configurations))
else:
    print(pd.DataFrame(base_configurations).to_string())

# <markdowncell>
# Atributos principais a seram variados e seus valores

# <codecell>
variations = [
    ('learning_rate_init', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
    ('hidden_layer_sizes', [(2 ** x,) for x in range(1, 8)]),
    ('max_iter', [200, 400, 800, 1600, 3200, 6400]),
    ('learning_rate', ['constant', 'invscaling', 'adaptive'])
]

if NOTEBOOK:
    display(pd.DataFrame(variations))
else:
    print(pd.DataFrame(variations).to_string())

# <markdowncell>
# Testes individuais com as diferentes bases geradas com os algoritmos de sampling
# * K-Means

# <codecell>
configuration_range = 3

for sample_method, sample_folds in samples_folds.items():
    print(f'optimize model configuration for {sample_method} with configuration range of {configuration_range}')
    configurations = base_configurations
    results = None
    for i, (variation_name, variation_values) in enumerate(variations):
        print(f'variation {i} -> {variation_name}')
        configurations_results = validate_model_configurations_variations(model_class, configurations,
                                                                          variation_name, variation_values,
                                                                          samples_folds['k-means'], base_folds,
                                                                          verbose=1)

        db_frame = pd.DataFrame([configuration for configuration, result in configurations_results])
        db_frame['mean score'] = [np.mean(result['score']) for _, result in configurations_results]
        db_frame['mean mse'] = [np.mean(result['mse']) for _, result in configurations_results]
        db_frame['mean roc auc'] = [np.mean(result['roc']['auc']) for _, result in configurations_results]
        db_frame['data'] = [(c, r) for c, r in configurations_results]
        db_frame.drop_duplicates([column for column in db_frame if column != 'data'], inplace=True)

        db_frame.sort_values(by=['mean roc auc', 'mean score'], ascending=[False, False], inplace=True)
        filtered_data_frame = db_frame[[column for column in db_frame if column != 'data']]
        if NOTEBOOK:
            display(filtered_data_frame)
        else:
            print(filtered_data_frame.to_string())

        configurations = [c for c, _ in db_frame['data'][:min(configuration_range, len(db_frame['data']))]]
        results = [r for _, r in db_frame['data'][:min(configuration_range, len(db_frame['data']))]]
        plot_multi_configuration_roc_curves((r['roc'] for r in results), db_frame.index)
        print()

    print('best configuration')
    best_configuration = configurations[0]
    best_configuration_results = results[0]
    plot_single_configuration_roc_curves(best_configuration_results['roc'],
                                         title='Best configuration for k-means ROC Curve')
    plot_single_configuration_confusion_matrix(best_configuration_results['matrix'], ['neg', 'pos'],
                                               title='Best configuration for k-means Confusion Matrix')

    print('testing overfiting (increased verbose)')
    overfiting_configuration = best_configuration.copy()
    overfiting_configuration['max_fail'] = 1000
    overfiting_configuration['max_iter'] = 30000
    overfiting_configuration_results = validate_model(model_class(**overfiting_configuration), samples_folds['k-means'],
                                                      base_folds, verbose=3)
    plot_single_configuration_roc_curves(overfiting_configuration_results['roc'],
                                         title='Best configuration with overfiting for k-means ROC Curve')
    plot_single_configuration_confusion_matrix(overfiting_configuration_results['matrix'], ['neg', 'pos'],
                                               title='Best configuration with overfiting for k-means Confusion Matrix')
    print()
    print()
    print()
