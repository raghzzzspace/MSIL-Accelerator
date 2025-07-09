import matplotlib.pyplot as plt
import pandas as pd

def hyperparams(model_name):
    if model_name == 'LogisticRegression':
        return {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'fit_intercept': [True, False],
            'max_iter': [100, 200, 500],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        }

    if model_name == 'DecisionTreeClassifier':
        return {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'ccp_alpha': [0.0, 0.01, 0.1],
            'random_state': [42]
        }

    if model_name == 'RandomForestClassifier':
        return {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        }

    if model_name == 'GradientBoostingClassifier':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 1.0],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt', 'log2'],
            'random_state': [42]
        }

    if model_name == 'XGBClassifier':
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.5],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5],
            'use_label_encoder': [False],
            'eval_metric': ['logloss'],
            'random_state': [42]
        }

    if model_name == 'GaussianNB':
        return {
            'var_smoothing': [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]
        }

    if model_name == 'MultinomialNB':
        return {
            'alpha': [0.0, 0.1, 0.5, 1.0],
            'fit_prior': [True, False]
        }

    # if model_name == 'BernoulliNB':
    #     return {
    #         'alpha': [0.0, 0.1, 0.5, 1.0],
    #         'binarize': [0.0, 0.5, 1.0],
    #         'fit_prior': [True, False]
    #     }

    if model_name == 'LinearSVC':
        return {
            'C': [0.1, 1, 10],
            'tol': [1e-3, 1e-4],
            'class_weight': [None, 'balanced'],
            'max_iter': [1000, 5000]
        }

    if model_name == 'SVC':
        return {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'tol': [1e-3, 1e-4],
            'class_weight': [None, 'balanced'],
            'max_iter': [1000, 5000]
        }

    if model_name == 'VotingClassifier':
        return {
            'voting': ['hard', 'soft']
        }

    if model_name == 'BaggingClassifier':
        return {
            'n_estimators': [10, 50, 100],
            'max_samples': [0.5, 0.7, 1.0],
            'max_features': [0.5, 0.7, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'random_state': [42]
        }

    if model_name == 'AdaBoostClassifier':
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R'],
            'random_state': [42]
        }

    # if model_name == 'MLPClassifier':
    #     return {
    #         'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    #         'activation': ['relu', 'tanh'],
    #         'solver': ['adam', 'sgd'],
    #         'alpha': [0.0001, 0.001],
    #         'learning_rate': ['constant', 'adaptive'],
    #         'max_iter': [200, 500],
    #         'random_state': [42]
    #     }

    # Safety fallback
    return {}

def visualize_results(search_obj, param_name, buffer):
    """
    Visualizes how a single hyperparameter affects the model's mean cross-validation score.

    Parameters:
        search_obj: The fitted GridSearchCV or RandomizedSearchCV object.
        param_name (str): The name of the hyperparameter to visualize (e.g., 'max_depth').

    Raises:
        ValueError: If the specified parameter is not found in the search results.

    Displays:
        A line plot showing the relationship between the hyperparameter values and mean CV scores.
    """
    results = pd.DataFrame(search_obj.cv_results_)

    # Handle case where 'param_' prefix is added in cv_results_
    # if param_name not in results.columns and f'param_{param_name}' in results.columns:
    #     param_name = f'param_{param_name}'

    # # Raise error if parameter still not found
    # if param_name not in results.columns:
    #     raise ValueError(f"Parameter '{param_name}' not found in search results.")

    if f'param_{param_name}' in results.columns:
        param_name = f'param_{param_name}'
    elif param_name in results.columns:
        pass
    else:
        print("Available columns:", results.columns.tolist())
        raise ValueError(f"Parameter '{param_name}' not found in search results.")
    # Plot mean CV score vs parameter
    plt.figure(figsize=(8, 5))
    plt.plot(results[param_name], results['mean_test_score'], marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Performance vs {param_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buffer, format = 'png')

import json

def logger(model_name: str, best_params, best_score=None, save_path=None):
    """
    Logs and optionally saves the best hyperparameter tuning results.

    Parameters:
        model_name (str): Name of the model (e.g., 'RandomForestClassifier').
        best_params (dict): Dictionary of the best hyperparameters found.
        best_score (float, optional): The best cross-validation score achieved.
        save_path (str, optional): If provided, the log will be saved to this file path as JSON.

    Prints:
        A summary of the model name, best parameters, and optionally the best score.

    Saves:
        The log to a JSON file if `save_path` is provided.
    """
    log = {
        'model': model_name,
        'best_params': best_params
    }

    if best_score is not None:
        log['best_score'] = best_score

    print(f"Tuning Summary for {model_name}")
    print(json.dumps(log, indent=4))

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(log, f, indent=4)
        print(f"\n Log saved to: {save_path}")
