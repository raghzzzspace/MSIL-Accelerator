
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.mixture import GaussianMixture

def get_model(name, task='clustering'):
    """
    Returns an instance of a clustering model based on its name.

    Args:
        name (str): The short name of the desired model.
                    Expected values: 'kmeans', 'dbscan', 'gmm', 'birch'.
        task (str): The type of machine learning task. Defaults to 'clustering'.

    Returns:
        An unfitted scikit-learn model instance, or None if the name is not found.
    """
    if task != 'clustering':
        return None

    models = {
        'kmeans': KMeans(),
        'dbscan': DBSCAN(),
        'gmm': GaussianMixture(),
        'birch': Birch()
    }
    
    # .get(name) will return the model instance or None if the key doesn't exist
    return models.get(name.lower())

# Example usage (for testing purposes):
if __name__ == '__main__':
    kmeans_model = get_model('kmeans')
    print(f"Requesting 'kmeans': {kmeans_model}")
    
    dbscan_model = get_model('DBSCAN') # Should handle case-insensitivity
    print(f"Requesting 'DBSCAN': {dbscan_model}")
    
    unknown_model = get_model('agglomerative')
    print(f"Requesting 'agglomerative': {unknown_model}")

    regression_model = get_model('kmeans', task='regression')
    print(f"Requesting 'kmeans' for regression task: {regression_model}")