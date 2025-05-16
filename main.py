from FewGAD import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

"""
FewGAD-main.py
"""

params_grid = {
    'dataset': ['cora'], # cora citeseer BlogCatalog ACM Flickr Facebook
    'lr': [1e-3],
    'num_epoch': [200], #
    'batch_size': [200],
    'auc_test_rounds': [256],
    't': [15], # subgraph size
    'k': [3], # k-order
    # hyparameter
    'alpha': [0.3],
    'beta': [0.1],
    'few_size': [10], # few-shot size
    'gamma': [0.7],
    'seed': [1]
}

print(f"dataset:{params_grid['dataset']}")

best_score = -1
best_params = {}
results = []
runs = 1

for dataset in params_grid['dataset']:
    for lr in params_grid['lr']:
        for num_epoch in params_grid['num_epoch']:
            for batch_size in params_grid['batch_size']:
                for auc_test_rounds in params_grid['auc_test_rounds']:
                    for t in params_grid['t']:
                        for k in params_grid['k']:
                            for alpha in params_grid['alpha']:
                                for beta in params_grid['beta']:
                                    for few_size in params_grid['few_size']:
                                        for gamma in params_grid['gamma']:
                                            for seed in params_grid['seed']:

                                                print(f"runs:{runs}")
                                                runs += 1
                                                auc = train_model(dataset, lr, num_epoch, batch_size,auc_test_rounds, t, k, alpha, beta, few_size, gamma, seed)
                                                # print(f"best score: {best_score}")
                                                results.append(
                                                    {
                                                        'dataset': dataset,
                                                        'lr': lr,
                                                        'num_epoch': num_epoch,
                                                        'batch_size': batch_size,
                                                        'auc_test_rounds': auc_test_rounds,
                                                        't': t,
                                                        'k': k,
                                                        'alpha': alpha,
                                                        'beta': beta,
                                                        'few_size': few_size,
                                                        'gamma': gamma,
                                                        'seed': seed,
                                                        'auc': auc
                                                    }
                                                )
                                                if auc > best_score:
                                                    best_score = auc
                                                    # print(f"best score: {best_score}")
                                                    best_params = {
                                                        'dataset': dataset,
                                                        'lr': lr,
                                                        'num_epoch': num_epoch,
                                                        'batch_size': batch_size,
                                                        'auc_test_rounds': auc_test_rounds,
                                                        't': t,
                                                        'k': k,
                                                        'alpha': alpha,
                                                        'beta': beta,
                                                        'seed': seed,
                                                        'few_size': few_size,
                                                        'gamma': gamma
                                                    }
                                                print(f"best score:{best_score}")

print('Best score:', best_score)
print('Best params:', best_params)
