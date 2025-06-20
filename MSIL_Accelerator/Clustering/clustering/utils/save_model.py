# accelerator/Clustering/utils/save_model.py

import joblib
import os

def save_model(model, model_name, path='artifacts/clustering/'):
    """
    Saves a fitted model object to a specified path using joblib.

    This function is identical in structure and purpose to the regression
    equivalent. It can save any Python object, including fitted clustering models.
    The default path is changed to keep clustering artifacts separate.

    Args:
        model: The fitted model object to save.
        model_name (str): The name for the saved file (without extension).
        path (str): The directory path to save the model in.

    Returns:
        str: The full path to the saved model file.
    """
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, f"{model_name}.pkl")
    
    # Save the model object
    joblib.dump(model, file_path)
    
    print(f"INFO: Model saved successfully to {file_path}")
    
    return file_path

# Example usage (for testing purposes)
if __name__ == '__main__':
    from sklearn.cluster import KMeans
    import numpy as np

    # 1. Create and fit a sample clustering model
    sample_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans_model = KMeans(n_clusters=2, n_init='auto', random_state=42)
    kmeans_model.fit(sample_data)

    # 2. Use the save_model function
    saved_path = save_model(model=kmeans_model, model_name="sample_kmeans_model")

    # 3. Verify the file exists
    if os.path.exists(saved_path):
        print(f"VERIFICATION: File '{saved_path}' created.")
        
        # Optional: Load it back to confirm
        loaded_model = joblib.load(saved_path)
        print(f"VERIFICATION: Model loaded successfully. Type: {type(loaded_model)}")
        # Clean up the created artifact
        os.remove(saved_path)
        if not os.listdir(os.path.dirname(saved_path)):
            os.rmdir(os.path.dirname(saved_path))