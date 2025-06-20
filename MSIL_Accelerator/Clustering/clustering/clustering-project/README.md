# Clustering Project

This project implements a clustering model using KMeans and provides functionality to save the trained model to a specified path. The main utility of this project is the `save_model` function, which ensures that the model can be easily stored and retrieved for future use.

## Files

- `utils/save_model.py`: Contains the `save_model` function that saves a fitted model object using joblib. It includes an example of how to create and save a sample KMeans clustering model.
- `requirements.txt`: Lists the dependencies required for the project.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Import the `save_model` function from the `utils` module.
2. Create and fit a KMeans clustering model using your dataset.
3. Call the `save_model` function to save the trained model.

Example:

```python
from utils.save_model import save_model
from sklearn.cluster import KMeans
import numpy as np

# Create and fit a sample clustering model
sample_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans_model = KMeans(n_clusters=2, n_init='auto', random_state=42)
kmeans_model.fit(sample_data)

# Save the model
saved_path = save_model(model=kmeans_model, model_name="sample_kmeans_model")
```

## License

This project is licensed under the MIT License.