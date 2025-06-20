# accelerator/Clustering/utils/metrics.py

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def clustering_metrics(X, labels):
    """
    Calculates intrinsic evaluation metrics for a clustering result.

    Args:
        X (np.array or pd.DataFrame): The data that was used for clustering.
        labels (np.array): The cluster labels assigned by the algorithm.

    Returns:
        dict: A dictionary of clustering scores.
    """
    # Check if metrics can be computed (requires more than 1 cluster)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if num_clusters <= 1:
        return {
            'silhouette_score': 'N/A',
            'calinski_harabasz_score': 'N/A',
            'davies_bouldin_score': 'N/A',
            'num_clusters_found': num_clusters
        }

    return {
        'silhouette_score': silhouette_score(X, labels),
        'calinski_harabasz_score': calinski_harabasz_score(X, labels),
        'davies_bouldin_score': davies_bouldin_score(X, labels),
        'num_clusters_found': num_clusters
    }