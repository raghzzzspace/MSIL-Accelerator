# accelerator/Clustering/utils/data_prep.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df):
    """
    Selects numeric columns from a DataFrame and applies StandardScaler.
    
    This is the direct structural equivalent of the train_test_split_data function,
    acting as a simple wrapper for a standard preprocessing step.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - scaled_data (np.array): The scaled numeric data. Returns None if no numeric data.
            - scaler (StandardScaler): The fitted scaler object. Returns None if no numeric data.
    """
    # Select only the numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        return None, None
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    return scaled_data, scaler

# Example usage
if __name__ == '__main__':
    data = {
        'age': [25, 30, 35, 45, 50],
        'income': [50000, 60000, 75000, 90000, 120000],
        'category': ['A', 'B', 'A', 'C', 'B']
    }
    sample_df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(sample_df)
    
    scaled_features, fitted_scaler = scale_numeric_features(sample_df)
    
    if scaled_features is not None:
        print("\nScaled Numeric Features:")
        print(scaled_features)
        print("\nFitted Scaler Object:")
        print(fitted_scaler)