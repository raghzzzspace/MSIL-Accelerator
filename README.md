# MSIL_Accelerator
MSIL_Accelerator is a no-code Machine Learning platform developed for digital enterprises (specifically, MSIL). It provides end-to-end data analysis and modeling capabilities, including data upload, cleaning, exploratory analysis, feature engineering, model training, and prediction — all via a web interface. The platform’s functionality is organized into several key modules and components, each implemented in Python. This documentation explains each part of the system, its purpose, and how to use it, in detail.

## Table of Contents
- Installation & Requirements
- Overview of Modules
  1. Exploratory Data Analysis (EDA)
  2. Regression Module
  3. Classification Module
  4. Clustering Module
  5. Time Series Module
- Using the Platform (How-To)
- Limitations & Known Issues
- Future Directions

## Installation & Requirements
MSIL_Accelerator is a Python 3 application built with Flask for the backend and React for the frontend. To run it, you need a Python environment with the following libraries (as inferred from the code imports):
- Web & I/O: Flask (for the HTTP server), requests-toolbelt (for forming multipart responses).
- Data Handling: Pandas and NumPy for data manipulation and numerical operations.
- Visualization: Plotly, Matplotlib and Seaborn are used in the time-series EDA and tuning routines (e.g. plotting rolling stats, ADF tests, elbow/AIC plots).
- Machine Learning: scikit-learn (for preprocessing, models, metrics, feature engineering, and train/test splits), XGBoost for gradient boosting models, Prophet for time-series forecasting. Additional libraries include category_encoders for advanced categorical encoding and scipy for statistical methods (e.g. z-score, linkage clustering).
- Utilities: joblib for saving models, statsmodels for ARIMA, plotly (used on the frontend)

Refer to the requirements.txt for a full list of libraries to import

## Overview of Modules
The MSIL_Accelerator codebase is organized into several Python modules, each with specific functionality:
- EDA Module (eda.py): Exploratory data analysis and data preprocessing (cleaning, feature engineering).
- Regression Module: Supports training various regression models. Key submodules include model factories, hyperparameter configurations, and utilities.
- Classification Module: Analogous to regression but for classification models.
- Clustering Module: Unsupervised clustering models and related utilities.
- Time Series Module (TS/TS.py): Special tools for time-series analysis and forecasting (e.g. ARIMA, Prophet).
- Web App (app.py): Ties everything together in a Flask web application, defining HTTP routes for uploading data, running analyses, training models, and returning results to the frontend.

Below, each module is described in detail.
1. Exploratory Data Analysis (EDA):
  The EDA class in eda.py provides functions for exploring and cleaning data.
  Key functionalities include:
  1. Data Overview (describe): Given a Pandas DataFrame, describe() returns summary information including column names, shape, summary statistics, duplicate count, missing value counts, and data types. For example, the describe method returns a JSON containing cols, shape, descriptive stats, duplicate count, null counts, and types
GitHub
  2. Univariate Analysis (univariate_analysis): This method generates basic summaries or plot data for a single column. Depending on the column’s type (categorical or numerical) and a requested plot type, it returns data for:
    - Categorical: Countplot or Piechart (value counts and percentages)
    - Numerical: Histogram, Distribution plot (KDE), or Boxplot (bin counts, sorted values, or quartiles with detected outliers)

  3. Multivariate Analysis (multivariate_analysis): Provides data for different multi-column plots, given selected columns and plot type:
      - Pairplot: Returns rows of data for plotting pairwise scatterplots among multiple numeric columns
      - Scatter/Line plots for two numerical columns
      - Barplot for one numeric vs one categorical (group means)

  4. Missing Value Imputation (fill_missing_values): Imputes missing data in one column, based on the column type (numerical or categorical) and method. Supported strategies are:
    - Drop row: removes any rows where the column is null.
    - Numerical: fill with mean, median, an arbitrary constant, random selection, end-of-distribution (mean+3σ), k-NN imputation, or iterative imputation
    - Categorical: fill with mode, label "Missing", forward/backward fill, random, a missing indicator column, k-NN (via coded category), or iterative imputation
     
  5. Outlier Handling (remove_outliers): Removes or caps outliers in a numeric column. Options:
Z-Score: Compute z-scores and either trim (drop rows) or cap (cap at threshold) where |z| > threshold
GitHub
.
IQR: Use interquartile range (IQR). Values outside Q1 – k*IQR or Q3 + k*IQR are trimmed or capped
GitHub
.
Percentile: Specify percentile bounds (e.g. 1st and 99th); trim or cap beyond those percentiles
GitHub
.
If method is 'NA', the function returns the data unchanged
GitHub
.
Feature Encoding (feature_encoding): Transforms individual features:
Numerical:
Discretization (binning): Unsupervised (uniform, quantile, k-means) or supervised (using a decision tree to create bins)
GitHub
.
Binarization: Creates binary column based on threshold (0 by default)
GitHub
.
Categorical:
Ordinal encoding (input): Map categories to integer codes (ordinal encoder)
GitHub
.
Ordinal encoding (output): Label-encode based on a target, if provided
GitHub
.
Nominal (One-Hot): One-hot encode, dropping the first category and expanding into new columns
GitHub
.
The new encoded columns are added to the DataFrame (original column kept or dropped as shown).
Feature Scaling (feature_scaling): Scales numeric features. Options:
Standardization (Z-score): Subtract mean and divide by std
GitHub
.
Normalization:
Min-max: scale to [0,1]
GitHub
.
Mean norm, max-abs, robust: apply other normalizations (e.g. L2 norm across rows, absolute max, or robust scaling)
GitHub
.
The result is appended as a new column (e.g. column_zscore, column_minmax) in the DataFrame.
Mixed Data Handling (handle_mixed_data): Splits columns with mixed types:
Type1 (e.g. 'C45'): Splits each string into alphabetic prefix and numeric suffix, creating two new columns [col]_cat and [col]_num
GitHub
.
Type2 (alternating): If a column alternates alphabetic and numeric entries down rows, it produces two new columns where one has only the letters and the other only numbers, aligning them on new rows
GitHub
.
Column Splitting (split_based_on_delimiter): Splits a string column by a given delimiter (default '-') into multiple parts. For example, splitting "A-B-C" into new columns "col_part1", "col_part2", etc
GitHub
.
Feature Transformation (feature_transformation): Applies a mathematical transformation to a column:
Type1='function': apply log, reciprocal, square, or square-root to the column values
GitHub
.
Type1='power': apply Box-Cox or Yeo-Johnson power transform (requires positive data for Box-Cox)
GitHub
.
The column is transformed in place (no new columns are created).
Manual Feature Selection (manual_feature_selection): Keeps only a specified subset of features (plus the target). Given a list of columns and a target, it returns a DataFrame with only those columns
GitHub
.
Feature Selection/Extraction (feature_selection_extraction): Either selects the top n features or extracts components:
Selection: Uses a LogisticRegression base model with forward or backward sequential selection to pick n_features from X (with respect to a target)
GitHub
. The output is a reduced feature set X_new.
Extraction: Applies PCA, LDA, or t-SNE to reduce features to n_components=n_features
GitHub
. PCA and LDA produce new numeric components, while t-SNE also provides n_features components (though note: t-SNE is typically non-deterministic).
In summary, the EDA module equips the platform with rich data exploration and preprocessing tools. For example, after uploading a dataset, the user can compute summary statistics, explore distributions of columns, remove outliers, impute missing values with various strategies, encode categorical variables, scale features, and derive new features — all via the web interface buttons and forms. These methods return Python dictionaries or modified DataFrames as JSON, which the front end uses to plot charts or update the dataset.
3. Regression Module
The Regression component supports training common regression models on a labeled dataset. Its key parts:
Model Factory (Regressor/models/trainer.py): Defines a function get_model(name) that returns an unfitted scikit-learn regressor by name
GitHub
. Available models are:
Linear Regression ('linear')
Ridge Regression ('ridge')
Lasso Regression ('lasso')
Decision Tree Regressor ('decision_tree')
Random Forest Regressor ('random_forest')
Support Vector Regressor ('svr')
XGBoost Regressor ('xgb')
GitHub
.
The web UI allows the user to select among these models (see Frontend Components below).
Hyperparameter Grids (Regressor/tuning/hyperparam.py): Provides predefined hyperparameter ranges for each model when tuning is requested. For example, for Ridge regression it defines alpha, max_iter, solver, etc.
GitHub
. These grids are used in GridSearchCV or RandomizedSearchCV in the training process. There is also a visualize_results function that plots CV results for a chosen parameter, and a logger that prints summary of tuning (best params, scores)
GitHub
GitHub
.
Data Splitting: A utility function splits features and target into train/test sets (20% by default). In the code (in app.py), upon training, the data for regression (df['regression']) is split with 80/20 ratio
GitHub
.
Training Pipeline (in app.py):
Data Retrieval: The global df['regression'] holds the uploaded DataFrame. The target column is chosen by the user via the UI form (JSON payload data['target'])
GitHub
.
Train/Test Split: Uses the utility train_test_split_data (stratified=false for regression)
GitHub
.
Model Selection: The user-selected model name is passed; the code actually instantiates a default model (currently Ridge in code snippet) and retrieves the hyperparameter grid for that model.
Hyperparameter Tuning: If the user requested tuning (data['tuning'] is "GridSearchCV" or "RandomizedSearchCV"), the code wraps the model in an sklearn search (GridSearchCV or RandomizedSearchCV) over the grid for the appropriate scoring (R^2 by default for regression)
GitHub
GitHub
. The search object’s best estimator (tuned_model) is obtained; if no tuning is requested, the code simply uses the original model.
Result Visualization: After tuning, the code generates a plot of performance vs. a key parameter and encodes it in the response (as a PNG image)
GitHub
.
Model Fitting and Evaluation: The chosen (possibly tuned) model is fitted on the training data, then predictions on the test set are made. Regression metrics (Mean Absolute Error, Mean Squared Error, R²) are computed by the helper regression_metrics (imported but not shown; presumably returns a dictionary of 'MAE', 'MSE', 'R2'). These metrics are added to a counter and included in the JSON response
GitHub
.
Output: The Flask route /train/regression returns a multipart/form-data response containing: (a) a JSON field "metrics" with the regression metrics and run number, and (b) an image file "regGraph.png" showing the tuning plot
GitHub
. The frontend displays the numeric results and graph to the user.
Prediction: Once a model has been trained and stored in memory (models['regression']), the user can run it on new data via /run/regression
GitHub
. The user uploads a CSV of feature values; the code reads it into a DataFrame, applies predict(), appends a "Predicted Values" column, and sends back a CSV with predictions.
In summary, the Regression module allows a user to upload labeled data, choose a target and model, optionally tune hyperparameters, and receive error metrics and a diagnostic plot. All supported models and their tuning parameters are listed above, and the interface ensures the process requires no coding from the user.
4. Classification Module
The Classification component is analogous to Regression, but for classification tasks. It includes:
Model Factory (Classification/models/trainer.py): The function get_model(name) returns a scikit-learn classifier by key
GitHub
. Available classifiers:
Logistic Regression ('logistic')
Decision Tree ('decision_tree')
Random Forest ('random_forest')
Gradient Boosting ('gradient_boosting')
XGBoost ('xgb')
Gaussian/Multinomial/Bernoulli Naive Bayes ('gaussian_nb', 'multinomial_nb', 'bernoulli_nb')
Support Vector Machines ('linear_svm', 'kernel_svm')
Voting Classifier ('voting') combining LR, RF, SVC (soft vote)
Bagging and AdaBoost (ensemble of trees)
MLP Neural Network ('ann' with MLPClassifier)
GitHub
.
The web UI lets the user pick among these (see the frontend component).
Hyperparameter Tuning: A separate hyperparameter grid for classification models is implied (the example code imports Classification.tuning.hyperparam). For each model, reasonable parameter choices would be defined (e.g. tree depth, number of estimators, regularization C, etc.). The training pipeline (in app.py) uses these similarly to regression.
Data Splitting: The code uses train_test_split_data with stratification to preserve class ratios.
Training Pipeline (in app.py): Very similar to regression:
Data & Target: df['classification'] holds the uploaded dataset; user selects the target column from it.
Split: Stratified train/test split (80/20).
Model & Tuning: Instantiate selected classifier, retrieve hyperparam grid, run GridSearchCV/RandomizedSearchCV with 'accuracy' (or other) as scoring
GitHub
GitHub
.
Fit & Evaluate: Fit on training set, predict on test set, compute classification metrics (accuracy, precision, recall, F1) via classification_metrics
GitHub
. This function returns a dict with those four scores
GitHub
.
Response: Returns a multipart response containing "metrics" (JSON of the scores and run count) and an image "classGraph.png" (tuning plot for a key parameter)
GitHub
.
Prediction: Similar to regression, the /run/classification route reads an input CSV, runs predict(), appends a "Predicted Values" column, and sends the CSV back
GitHub
.
The frontend form for Classification (see static/components/classification.js) guides the user through uploading data, selecting model/tuning, and then displays a table of metrics and the prediction output download. The classification metrics logged match those from classification_metrics
GitHub
.
5. Clustering Module
The Clustering component handles unsupervised learning (grouping) and includes:
Model Factory (Clustering/clustering/models/trainer.py): get_model(name) returns a clustering model instance
GitHub
. Supported names:
kmeans (sklearn.cluster.KMeans)
dbscan (sklearn.cluster.DBSCAN)
gmm (sklearn.mixture.GaussianMixture)
birch (sklearn.cluster.Birch)
If an unknown name is given, it returns None
GitHub
.
Hyperparameters (Clustering/clustering/tuning/hyperparam.py): Defines parameter grids for tuning clustering algorithms:
k-means: range of n_clusters, init, n_init, etc.
GitHub
.
DBSCAN: eps, min_samples, distance metric, etc.
GitHub
.
GMM: n_components, covariance_type, etc.
GitHub
.
BIRCH: threshold, branching_factor, n_clusters
GitHub
.
There is also a function to visualize cluster evaluation: for k-means/BIRCH it runs an elbow method on inertia, for GMM it plots AIC/BIC over components
GitHub
GitHub
. A tune_model function performs Grid/Randomized search optimizing silhouette score
GitHub
GitHub
.
Metrics (Clustering/clustering/utils/metrics.py): After fitting, it computes intrinsic clustering scores: Silhouette, Calinski-Harabasz, and Davies-Bouldin
GitHub
. It also returns the number of clusters found. If only one cluster is found, it reports 'N/A' for those scores
GitHub
.
Training Workflow (in app.py):
Data: df['clustering'] holds the uploaded data (unsupervised, so all features)
GitHub
.
Model & Tuning: Based on data['model'], obtain the model and param grid. Call ClusteringHyperparameters.tune_model(...) (from the clustering hyperparam file) to get a tuned model and logs
GitHub
. This uses silhouette-based scoring under the hood
GitHub
.
Fit: Fit the tuned model on the data, get cluster labels from labels_
GitHub
.
Evaluate: Compute clustering metrics on the entire dataset with labels
GitHub
.
Output: Append the cluster labels to the data and send it back as a CSV (clusters.csv) in the response, along with the metrics JSON (including run count) and an evaluation graph (clusterGraph.png) if generated
GitHub
. The user can then download a CSV of the original data with an added “assigned_cluster” column.
For example, running k-means via the run_clustering.py script shows this process: it scales data, fits k-means, computes silhouette, adds assigned_cluster, logs the score, and saves the model
GitHub
GitHub
. The web app automates this via the REST API.
6. Time Series Module
The Time Series module (TS/TS.py) provides functions and models for time-series forecasting:
Date Feature Engineering:
add_datetime_features(df, datetime_col): Given a datetime column, extracts year, month, day, weekday, week-of-year, and indicators for weekend, month start/end, quarter start/end, year start/end
GitHub
. These features can help models capture seasonality or trends.
add_lag_features(df, target_col, lags=[1,7,14]): Creates lagged versions of the target variable (shifted by 1, 7, 14 days by default)
GitHub
.
Preprocessing (preprocess_time_series): Takes raw time-series data, ensures a uniform frequency (defaults to daily), interpolates missing target values, adds datetime and lag features, and drops any rows still containing NaNs
GitHub
. This results in a feature matrix with the original datetime, the lag columns, and the date-part features ready for modeling.
EDA (perform_time_series_eda): Generates plots and statistics for the series. It sorts by date, plots the original series, rolling mean and std, seasonal decomposition (if possible), and ACF/PACF. It also runs the Augmented Dickey-Fuller test to check stationarity, printing the statistic and p-value
GitHub
GitHub
GitHub
. (All plots are shown via Matplotlib/Seaborn, intended for the developer or user to inspect when running locally or logged.)
Models: The module defines custom estimator classes for time-series:
ARIMAModel: Wraps statsmodels’ ARIMA, but first automatically determines differencing needed via successive ADF tests. It fits on the transformed series and can forecast future points, handling differencing internally
GitHub
GitHub
.
ProphetModel: Wraps Facebook’s Prophet; it expects a DataFrame with columns ds (date) and y, then fits and predicts, returning the forecasted values
GitHub
.
LinearTimeSeriesModel: A simple linear regression on time index (for baseline)
GitHub
.
The helper get_model(name) returns one of these by key: 'arima', 'prophet', or 'linear'
GitHub
.
Forecasting Workflow (in app.py):
Data: df['timeseries'] is the uploaded dataset. The user specifies the datetime column and target column in JSON. The code copies the raw data and calls perform_time_series_eda for logging and plots (not directly returned to UI).
Preprocessing: The raw DataFrame is preprocessed via preprocess_time_series, creating lags and features.
Train/Test Split: The code uses a fixed test size (last 30 points). The features X are all columns except date and target; y is the target.
Model & Tuning: Fetch the selected model (ARIMA/Prophet/Linear). Get hyperparameters with hyperparams(model_name) (e.g., ARIMA order options)
GitHub
. If the training set is too small or no grid is provided, skip tuning; otherwise run Grid/Randomized CV.
Fit & Predict: Fit the final model on X_train, y_train, then predict on X_test
GitHub
.
Metrics & Output: Compute forecast metrics (Mean Absolute Error and RMSE) via forecast_metrics
GitHub
, package them (with a counter), and also save a plot of the forecast. Return these in a multipart response (metrics JSON and forecastGraph.png)
GitHub
.
Uploading Time Series: A dedicated route /upload/timeseries reads a CSV into df['timeseries'] and returns its columns
GitHub
.
Thus, the Time Series module gives the user tools to explore seasonality and stationarity, engineer time-related features, and train specialized forecasting models with tuning. The web UI’s Time Series tab likely provides a form to select date/target and model, then displays the error metrics and forecast plot after training.
Using the Platform (How-To)
Starting the App: The main web app is in app.py. Run it (e.g. python app.py); Flask will serve on localhost:5000 by default. The homepage redirects to the static React frontend (index.html). Frontend Interface: The UI (in static/) is a single-page React app. The sidebar has navigation:
EDA Supervised/Unsupervised (for data upload and cleaning),
Classification, Regression, Time Series, Clustering tabs
GitHub
.
Each tab loads a React component (e.g. <Regression/>, <Classification/>), which presents file upload fields, dropdowns for options, and buttons for actions. For example:
In Regression, the user uploads a training CSV, picks a model and split ratio, chooses tuning method, and clicks “Train”
GitHub
GitHub
. The resulting table of metrics and an image graph appear; then an output file field and a “Run” button allow uploading new input to generate predictions (downloaded as CSV)
GitHub
GitHub
.
In Classification, a similar form appears for classification models
GitHub
GitHub
.
Time Series and Clustering have analogous interfaces (not shown here but implemented in static/components/timeseries.js, clustering.js).
The EDA section (edasl.js and edaul.js, not shown) lets the user upload data and perform cleaning operations. For example, after uploading, the app will call /upload/edasl, get the columns and summary, and display them; then the user can specify how to impute missing values or remove outliers (the frontend collects configs and POSTs to /handle-missing or /remove-outliers), or perform encoding/scaling by POSTing JSON to /feature-encoding or /feature-scaling.
APIs: The backend provides REST endpoints (e.g. POST /upload/<model>, /train/<model>, /run/<model>, /univariate, /multivariate, etc.) which the React frontend calls via fetch. These endpoints correspond to the methods in app.py described above. For instance, uploading a regression dataset triggers uploadFile('regression') which hits /upload/regression and returns the column list for target selection
GitHub
. Training hits /train/regression or /train/classification etc. Working Example (Regression):
Prepare a CSV of training data with numeric features and one target column.
Go to the Regression tab, upload the file. The UI then lists all columns.
Select a model (e.g. “Random Forest Regression”), choose tuning (Grid or Random), set train-test split, and pick the target column from the dropdown.
Click Train. The backend splits data, tunes and fits the model, computes MAE/MSE/R², and returns these metrics plus a graph image. The UI shows the scores and table.
Now in “Run”, upload a new CSV (with same feature columns, no target). Click Run: the backend predicts with the trained model and returns a CSV for download, containing original data plus “Predicted Values.”
This same pattern applies to Classification and Clustering (with outputs adapted for labels or clusters), and Time Series (where the “Run” action actually shows a plot of forecast). The EDA/feature-engineering steps are done via separate forms/buttons that call the preprocessing endpoints and update the dataset state in the app.
Limitations & Known Issues
No Official Issue Tracker Entries: There are no documented GitHub issues or bug reports in the repository to date. (A search of the repo’s issues returned no results, suggesting either a private issue tracker or none have been filed.)
Feature Gaps:
The classification hyperparameter grids are not explicitly defined in the code (the regression hyperparams file has many entries, but a separate classification hyperparams file is not visible). As a result, tuning for classification may use default or limited options.
The EDA module covers many methods, but some advanced analysis (e.g. correlation heatmap for numeric variables, multi-categorical plots) are not implemented.
The front end assumes datasets are clean CSV files with no missing values in target columns; while missing values can be imputed, the UI flow might break on unexpected data types.
The global data storage (df dictionary in app.py) means only one dataset per type can be handled at a time. Concurrent users would interfere with each other (there is no user/session isolation).
Performance & Scaling: The app performs all computation synchronously on the server. Very large datasets or heavy models (e.g. GridSearch on big data) may be slow or cause timeouts. There is no built-in paging or streaming of data.
Model Persistence: The code saves the last trained model in memory but does not persist it to disk or support loading of previously saved models (except in example scripts). Thus, after the server restarts, any trained models are lost. This also means you cannot train multiple models sequentially and choose among them later — only the most recent model of each type is kept.
Error Handling: Some endpoints have basic try/except, but error messages are often generic. For example, uploading a wrong file format may cause an exception. The frontend may not always catch or display detailed error info.
Future Directions
Potential extensions and improvements for MSIL_Accelerator include:
User & Session Management: Allow multiple users with separate workspaces or sessions, avoiding global state conflicts. Store intermediate data and trained models in a database or file system.
Model Repository: Add functionality to save trained models to disk (or a model registry) and load them later. The clustering project includes an example save_model function
GitHub
; similar features could be integrated for regression/classification models.
Expanded Algorithms: Incorporate more models (e.g. additional time-series methods, deep learning models, newer clustering algorithms) and corresponding hyperparameter grids. The commented-out hyperparameters (e.g. in RandomForest: 'monotonic_cst' or split methods) hint at possible future enhancements.
Enhanced Visualization: Improve plots in the UI. Currently, the backend returns static PNGs (e.g. from Matplotlib) for some tasks. Interactive or higher-quality charts (e.g. using Plotly) could improve user experience. For EDA plots, integrating dynamic charting on the frontend (e.g. histograms, boxplots) could make analysis smoother.
Automated Analytics: Implement features like automatic feature importance reporting, or automated model selection (e.g. train multiple models and pick the best).
Error Handling & Logging: More robust input validation (checking for nulls before modeling), user-friendly error messages, and persistent logging of actions would help in production use.
Performance Optimizations: For large datasets, techniques like incremental learning, sampling, or offloading heavy computations to background workers could be added.
Documentation & Tutorials: While this document covers developer-centric details, end-user guides (e.g. example data walkthroughs, video tutorials) would help non-technical users learn the platform more easily.
Given the current code, MSIL_Accelerator already provides a comprehensive no-code pipeline. Future work can build on this foundation to make it more robust, user-friendly, and scalable.
