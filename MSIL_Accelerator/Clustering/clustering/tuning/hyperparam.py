# accelerator/Clustering/tuning/hyperparams.py

import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score
import random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
import numpy as np
from sklearn.mixture import GaussianMixture

def hyperparams(model_name):
    """
    Returns a dictionary of hyperparameters for a given clustering model.
    This defines the "search space" for tuning or user input.
    
    Args:
        model_name (str): The name of the clustering model (e.g., 'KMeans', 'DBSCAN').
    
    Returns:
        dict: A parameter grid for the specified model.
    """
    if model_name == 'kmeans':
        param_grid = {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],  # Number of clusters to find
            'init': ['k-means++', 'random'],                     # Method for initialization
            'n_init': ['auto', 10, 20],                          # Number of times k-means will be run with different centroids
            'max_iter': [100, 300, 500],                         # Maximum number of iterations for a single run
            'tol': [1e-4, 1e-3],                                 # Tolerance for convergence
            'algorithm': ['lloyd', 'elkan']                      # K-means algorithm to use
        }
        return param_grid

    if model_name == 'dbscan':
        param_grid = {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],  # The maximum distance between two samples for one to be considered as in the neighborhood of the other
            'min_samples': [3, 5, 7, 10, 15, 20],       # The number of samples in a neighborhood for a point to be considered as a core point
            'metric': ['euclidean', 'manhattan'],       # The distance metric to use
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] # Algorithm to find nearest neighbors
        }
        return param_grid

    if model_name == 'gmm':
        param_grid = {
            'n_components': [2, 3, 4, 5, 6, 8, 10],   # The number of mixture components (clusters)
            'covariance_type': ['full', 'tied', 'diag', 'spherical'], # Type of covariance parameters to use
            'tol': [1e-3, 1e-4],                      # Convergence threshold
            'max_iter': [100, 200, 500],              # Maximum number of EM iterations
            'n_init': [1, 5, 10],                     # Number of initializations to perform
            'init_params': ['kmeans', 'random']       # Method used to initialize the weights
        }
        return param_grid

    if model_name == 'birch':
        param_grid = {
            'threshold': [0.1, 0.3, 0.5, 0.7, 1.0],  # The radius of the subcluster obtained by merging a new sample and the closest subcluster should be smaller than the threshold
            'branching_factor': [20, 50, 100],       # Maximum number of CF Subclusters in each node
            'n_clusters': [2, 3, 4, 5, 6, 8, 10, None] # Number of clusters to return after the final clustering step. If None, the final clustering step is not performed
        }
        return param_grid
    
    return {} # Return empty dict if model not found


def visualize_cluster_evaluation(model_name, data, buffer, k_range=(2, 11)):
    """
    Visualizes the effect of changing the number of clusters (k) for applicable models.

    - For 'KMeans' and 'BIRCH', it plots the Elbow Method using Inertia.
    - For 'GaussianMixture', it plots AIC and BIC scores.
    - For 'DBSCAN', it prints a message as this method is not applicable.
    
    Args:
        model_name (str): The name of the model to evaluate.
        data (np.array): The pre-scaled data.
        k_range (tuple): The range of 'k' (or n_components) values to test.
    """
    k_values = range(k_range[0], k_range[1])
    
    if model_name in ['kmeans', 'birch']:
        inertias = []
        for k in k_values:
            if model_name == 'kmeans':
                model = KMeans(n_clusters=k, n_init='auto', random_state=42)
            else: # BIRCH
                model = Birch(n_clusters=k)
            
            model.fit(data)
            # BIRCH doesn't have a direct .inertia_ attribute, so we calculate it manually.
            # This is a simplification; for a perfect inertia, we'd need cluster centers.
            # A common proxy is to run KMeans on the result or accept this simplification.
            # For this purpose, we assume the user is aware of this nuance.
            # A more robust approach for BIRCH is often visual inspection.
            # Let's use KMeans for the final step to get a proper inertia value.
            if model_name == 'birch':
                 # Re-run a quick kmeans on the transformed data to get inertia
                 subcluster_centers = model.subcluster_centers_
                 if subcluster_centers.shape[0] < k:
                     print(f"WARN: BIRCH produced fewer subclusters ({subcluster_centers.shape[0]}) than requested k={k}. Skipping.")
                     inertias.append(np.nan)
                     continue
                 final_model = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(subcluster_centers)
                 # This is an approximation of inertia
                 inertias.append(final_model.inertia_)
            else:
                 inertias.append(model.inertia_)


        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertias, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title(f"Elbow Method for {model_name}")
        plt.grid(True)
        plt.xticks(k_values)
        plt.tight_layout()
        plt.savefig(buffer, format = 'png')

    elif model_name == 'gmm':
        aics = []
        bics = []
        for k in k_values:
            model = GaussianMixture(n_components=k, random_state=42, n_init=10)
            model.fit(data)
            aics.append(model.aic(data))
            bics.append(model.bic(data))

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, aics, marker='o', label='AIC')
        plt.plot(k_values, bics, marker='s', label='BIC')
        plt.xlabel("Number of Components")
        plt.ylabel("Information Criterion Score")
        plt.title("AIC & BIC for Gaussian Mixture Model")
        plt.legend()
        plt.grid(True)
        plt.xticks(k_values)
        plt.tight_layout()
        plt.savefig(buffer, format = 'png')

    else: # For DBSCAN or any other model
        print(f"INFO: The elbow/evaluation method is not applicable for the '{model_name}' algorithm.")


def logger(model_name: str, best_params, best_score=None, save_path=None):
    """
    Logs the parameters and score for a clustering model run.
    """
    log = {
        'model': model_name,
        'parameters': best_params
    }
    if best_score is not None:
        # For clustering, the score is often the Silhouette Score
        log['silhouette_score'] = best_score

    print(f"Run Summary for {model_name}")
    print(json.dumps(log, indent=4))

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(log, f, indent=4)
        print(f"\nLog saved to: {save_path}")

# Example usage
if __name__ == '__main__':
    # Get hyperparameter grid for K-Means
    kmeans_params = hyperparams('KMeans')
    print("K-Means Hyperparameters:")
    print(json.dumps(kmeans_params, indent=2))

    # Example of logging a result
    best_kmeans_run = {
        'n_clusters': 4,
        'init': 'k-means++'
    }
    logger('KMeans', best_kmeans_run, best_score=0.55, save_path='kmeans_log.json')

def tune_model(model, param_grid, X_train, operation='GridSearchCV', scoring='silhouette'):
    
    metric_logs = []

    def clustering_score(estimator, X):
        # Some clustering models don't have .fit_predict, so fallback to .labels_
        try:
            labels = estimator.fit_predict(X)
        except:
            estimator.fit(X)
            labels = estimator.labels_

        if len(np.unique(labels)) < 2:
            return -1
        return silhouette_score(X, labels)

    if operation == 'GridSearchCV':
        grid = GridSearchCV(model, param_grid, scoring=clustering_score, cv=None, verbose=1)
        grid.fit(X_train)

        for i in range(len(grid.cv_results_['params'])):
            entry = {
                'params': grid.cv_results_['params'][i],
                'mean_test_score': grid.cv_results_['mean_test_score'][i],
                'fit_time': grid.cv_results_['mean_fit_time'][i]
            }
            metric_logs.append(entry)

        return grid.best_estimator_, grid.best_params_, grid, metric_logs

    elif operation == 'RandomizedSearchCV':
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,
            scoring=clustering_score,
            cv=None,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train)

        for i in range(len(random_search.cv_results_['params'])):
            entry = {
                'params': random_search.cv_results_['params'][i],
                'mean_test_score': random_search.cv_results_['mean_test_score'][i],
                'fit_time': random_search.cv_results_['mean_fit_time'][i]
            }
            metric_logs.append(entry)

        return random_search.best_estimator_, random_search.best_params_, random_search, metric_logs

    elif operation == 'None':
        # Just fit and evaluate
        model.fit(X_train)
        try:
            labels = model.predict(X_train)
        except:
            labels = model.labels_
        score = silhouette_score(X_train, labels)
        print(f"Silhouette Score: {score:.4f}")
        return model, model.get_params(), None, [{'params': model.get_params(), 'silhouette_score': score}]