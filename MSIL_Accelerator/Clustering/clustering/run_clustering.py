# run_clustering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

# Import the new clustering-specific modules
from models.trainer import get_model as get_cluster_model
from tuning.hyperparam import visualize_elbow_method, logger as cluster_logger

# --- 1. Data Loading and Preparation ---
# Using a sample dataset, you can replace this with your own.
# Let's use the same iris dataset but treat it as an unsupervised problem.
try:
    df = pd.read_csv('C:\\Users\\Ashok Kumar\\Desktop\\iris.csv')
    df = df.drop(['Id', 'Species'], axis=1, errors='ignore') # Drop non-numeric and label columns
except FileNotFoundError:
    print("WARN: Iris dataset not found. Using a dummy dataset.")
    from sklearn.datasets import make_blobs
    df, _ = make_blobs(n_samples=200, centers=4, n_features=4, random_state=42)
    df = pd.DataFrame(df, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])

# For clustering, we use all numeric features. There's no X/y split.
X_cluster = df.copy() 

# Scaling is crucial for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)


# --- 2. Model Selection and Configuration ---
# Choose a model to run
model_name = 'kmeans' 
model = get_cluster_model(model_name)

if model is None:
    raise ValueError(f"Model '{model_name}' not found in factory.")

# Set the parameters for this specific run.
# In a real app, these would come from the UI or a config file.
params = {'n_clusters': 4, 'n_init': 'auto', 'random_state': 42}
model.set_params(**params)


# --- 3. (Optional) Tuning Visualization for K-Means ---
# This is the equivalent of the `visualize_results` for regression
if model_name == 'kmeans':
    print("INFO: Performing Elbow Method analysis for K-Means...")
    visualize_elbow_method(X_scaled, k_range=(2, 11))


# --- 4. Model Training and Prediction ---
print(f"\nINFO: Fitting {model_name} model with parameters: {params}")
model.fit(X_scaled)
cluster_labels = model.labels_


# --- 5. Evaluation ---
# The primary metric for clustering is often the Silhouette Score.
# It requires both the data and the labels.
score = silhouette_score(X_scaled, cluster_labels)
print(f"INFO: Silhouette Score for the run: {score:.4f}")

# Add the cluster labels back to the original dataframe for inspection
X_cluster['assigned_cluster'] = cluster_labels
print("\nSample of data with assigned clusters:")
print(X_cluster.head())


# --- 6. Logging and Saving ---
# Use the logger to record the run's configuration and score
cluster_logger(model_name=str(model.__class__.__name__), best_params=params, best_score=score)

# Save the fitted model for future use
model_filename = f'{model_name}_model_fitted.pkl'
joblib.dump(model, model_filename)
print(f"\nINFO: Fitted model saved as '{model_filename}'")


# --- 7. Example of Loading and Predicting with the Saved Model ---
# Load the model and predict the cluster for new, unseen data
loaded_model = joblib.load(model_filename)

# Create some new data points (must have the same number of features)
new_data = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Similar to Iris-setosa
    [7.0, 3.2, 4.7, 1.4]   # Similar to Iris-versicolor
])

# IMPORTANT: New data must be scaled using the SAME scaler that was fit on the training data
new_data_scaled = scaler.transform(new_data) 

new_predictions = loaded_model.predict(new_data_scaled)
print(f"\nINFO: Predictions for new data points: {new_predictions}")