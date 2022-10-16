# File names
OTUS = "log10_relative_OTU_all.csv"
LABELS = "class_labels.csv"

# Hyper-parameter grid for training classifiers
svm_hyper_parameters = [{'C': [2 ** s for s in range(-4, 4, 1)], 'kernel': ['linear']},
                        {'C': [2 ** s for s in range(-4, 4, 1)], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}]
rf_hyper_parameters = [{'n_estimators': [2 ** s for s in range(7, 11, 1)],
                        'max_features': ['sqrt', 'log2'],
                        'criterion': ['gini', 'entropy']
                        }, ]
mlp_hyper_parameters = [{'hidden_layer_sizes': [(128, 64, 32), (128, 64, 32, 16), (128, 64, 32, 16, 8)],
                            'learning_rate': ['constant', 'invscaling', 'adaptive'],
                            'alpha': [10** s for s in range(-4, 0, 1)]
                            }]