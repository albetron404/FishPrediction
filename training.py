# This script is responsible for training machine learning models using the provided dataset and saving the best models along with their details.
# training.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from datetime import datetime
import os
import joblib
import json

def check_and_delete_log():
    """
    Check if train_log.json exists, and if it does, delete the existing one.

    Returns:
    None
    """
    if os.path.exists("static/train_log.json"):
        os.remove("static/train_log.json")
        print("Existing train_log.json deleted.")
    else:
        print("No existing train_log.json found.")

# Function to read dataset and perform initial data exploration
def read_data():
    """
    Read dataset and perform initial data exploration.

    Returns:
    df (DataFrame): DataFrame containing the dataset.
    """
    # Read the dataset
    df = pd.read_csv("Dataset/Fish.csv")

    # Data dictionary
    data_dict = {
        "Species": "Species of fish",
        "Weight": "Weight of fish in grams",
        "Length1": "Vertical length in cm",
        "Length2": "Diagonal length in cm",
        "Length3": "Cross length in cm",
        "Height": "Height in cm",
        "Width": "Width in cm"
    }

    # Print dataset description
    print("Dataset Description:")
    print(df.describe())

    # Identify and handle missing values
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("NA")

    return df

# Function to plot species-wise count
def plot_species_count(df):
    """
    Plot species-wise count.

    Args:
    df (DataFrame): DataFrame containing the dataset.
    """
    species_count = df['Species'].value_counts()
    species_count.plot(kind='bar', title='Species-wise Count')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.show()

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    """
    Plot correlation matrix.

    Args:
    df (DataFrame): DataFrame containing the dataset.
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    plt.title('Correlation Matrix')
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
    plt.show()

# Function to perform data preprocessing and split data for regression
def preprocess_and_split_regression(df):
    """
    Perform data preprocessing and split data for regression.

    Args:
    df (DataFrame): DataFrame containing the dataset.

    Returns:
    train_x_reg, test_x_reg, train_y_reg, test_y_reg: Train and test data for regression.
    """
    # One-hot encode species
    df = pd.get_dummies(df, columns=['Species'])

    # Split features and target variable
    X = df.drop(columns=['Weight'])
    y = df['Weight']

    # Split data into train and test sets
    train_x_reg, test_x_reg, train_y_reg, test_y_reg = train_test_split(X, y, test_size=0.2, random_state=42)

    return train_x_reg, test_x_reg, train_y_reg, test_y_reg

def preprocess_and_split_classification(df, scaler_filename='static/standard_scaler_model.pkl'):
    """
    Perform data preprocessing and split data for classification.

    Args:
    df (DataFrame): DataFrame containing the dataset.
    scaler_filename (str): Name of the file to save the scaler model.

    Returns:
    train_x_clas, test_x_clas, train_y_clas, test_y_clas: Train and test data for classification.
    """
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Serialize the StandardScaler model
    joblib.dump(scaler, scaler_filename)

    # Split features and target variable
    X = df.drop(columns=['Species'])
    y = df['Species']  # Target variable should represent the class labels for classification

    # Split data into train and test sets with stratified split
    train_x_clas, test_x_clas, train_y_clas, test_y_clas = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return train_x_clas, test_x_clas, train_y_clas, test_y_clas

# Modify train_regression_model function
def train_regression_model(train_x_reg, train_y_reg):
    """
    Train RandomForestRegressor model.

    Args:
    train_x_reg (DataFrame): Training features for regression.
    train_y_reg (Series): Training target variable for regression.

    Returns:
    reg_model: Trained regression model.
    """
    # Define hyperparameters grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Lists to store metric values for each run
    rmse_values = []

    # Iterate through hyperparameters grid
    for run_num, params in enumerate(ParameterGrid(param_grid), start=1):
        # Initialize RandomForestRegressor
        reg_model = RandomForestRegressor(**params, random_state=42)
        # Train the model
        reg_model.fit(train_x_reg, train_y_reg)
        # Make predictions
        reg_predictions = reg_model.predict(train_x_reg)
        # Evaluate model performance
        rmse = mean_squared_error(train_y_reg, reg_predictions, squared=False)
        # Store the RMSE value for the current run
        rmse_values.append(rmse)
        # Print run parameters and metric value
        print(f"Run {run_num}: Parameters: {params}, RMSE: {rmse}")

    # Plot RMSE values over grid search runs
    plt.plot(range(1, len(rmse_values) + 1), rmse_values, marker='o')
    plt.xlabel('Run Number')
    plt.ylabel('RMSE')
    plt.title('RMSE over Grid Search Runs (Regression)')
    plt.grid(True)
    plt.show()

    # Get the index of the run with the minimum RMSE
    best_run_index = np.argmin(rmse_values)
    best_params = list(ParameterGrid(param_grid))[best_run_index]

    # Re-train the model with the best parameters
    reg_model = RandomForestRegressor(**best_params, random_state=42)
    reg_model.fit(train_x_reg, train_y_reg)

    return reg_model


# Modify train_classification_model function
def train_classification_model(train_x_clas, train_y_clas):
    """
    Train RandomForestClassifier model.

    Args:
    train_x_clas (DataFrame): Training features for classification.
    train_y_clas (Series): Training target variable for classification.

    Returns:
    clas_model: Trained classification model.
    """
    # Define hyperparameters grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Lists to store metric values for each run
    accuracy_values = []

    # Iterate through hyperparameters grid
    for run_num, params in enumerate(ParameterGrid(param_grid), start=1):
        # Initialize RandomForestClassifier
        clas_model = RandomForestClassifier(**params, random_state=42)
        # Train the model
        clas_model.fit(train_x_clas, train_y_clas)
        # Make predictions
        clas_predictions = clas_model.predict(train_x_clas)
        # Evaluate model performance
        accuracy = accuracy_score(train_y_clas, clas_predictions)
        # Store the accuracy value for the current run
        accuracy_values.append(accuracy)
        # Print run parameters and metric value
        print(f"Run {run_num}: Parameters: {params}, Accuracy: {accuracy}")

    # Plot accuracy values over grid search runs
    plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, marker='o')
    plt.xlabel('Run Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Grid Search Runs (Classification)')
    plt.grid(True)
    plt.show()

    # Get the index of the run with the maximum accuracy
    best_run_index = np.argmax(accuracy_values)
    best_params = list(ParameterGrid(param_grid))[best_run_index]

    # Re-train the model with the best parameters
    clas_model = RandomForestClassifier(**best_params, random_state=42)
    clas_model.fit(train_x_clas, train_y_clas)

    return clas_model

# # Function to train RandomForestRegressor
# def train_regression_model(train_x_reg, train_y_reg):
#     """
#     Train RandomForestRegressor model.

#     Args:
#     train_x_reg (DataFrame): Training features for regression.
#     train_y_reg (Series): Training target variable for regression.

#     Returns:
#     reg_model: Trained regression model.
#     """
#     # Define hyperparameters grid for Random Forest
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }

#     best_metric = float('inf')
#     best_model = None

#     # Iterate through hyperparameters grid
#     for run_num, params in enumerate(ParameterGrid(param_grid), start=1):
#         # Initialize RandomForestRegressor
#         reg_model = RandomForestRegressor(**params, random_state=42)
#         # Train the model
#         reg_model.fit(train_x_reg, train_y_reg)
#         # Make predictions
#         reg_predictions = reg_model.predict(train_x_reg)
#         # Evaluate model performance
#         rmse = mean_squared_error(train_y_reg, reg_predictions, squared=False)
#         # Print run parameters and metric value
#         print(f"Run {run_num}: Parameters: {params}, RMSE: {rmse}")
#         # Check if current model outperforms the best model so far
#         if rmse < best_metric:
#             best_metric = rmse
#             best_model = reg_model

#     return best_model

# # Function to train RandomForestClassifier
# def train_classification_model(train_x_clas, train_y_clas):
#     """
#     Train RandomForestClassifier model.

#     Args:
#     train_x_clas (DataFrame): Training features for classification.
#     train_y_clas (Series): Training target variable for classification.

#     Returns:
#     clas_model: Trained classification model.
#     """
#     # Define hyperparameters grid for Random Forest
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }

#     best_metric = 0
#     best_model = None

#     # Iterate through hyperparameters grid
#     for run_num, params in enumerate(ParameterGrid(param_grid), start=1):
#         # Initialize RandomForestClassifier
#         clas_model = RandomForestClassifier(**params, random_state=42)
#         # Train the model
#         clas_model.fit(train_x_clas, train_y_clas)
#         # Make predictions
#         clas_predictions = clas_model.predict(train_x_clas)
#         # Evaluate model performance
#         accuracy = accuracy_score(train_y_clas, clas_predictions)
#         # Print run parameters and metric value
#         print(f"Run {run_num}: Parameters: {params}, Accuracy: {accuracy}")
#         # Check if current model outperforms the best model so far
#         if accuracy > best_metric:
#             best_metric = accuracy
#             best_model = clas_model

#     return best_model

# Function to save model and log model details
def save_model_and_log(model, model_type, evaluation_metric_value):
    """
    Save trained model and log model details.

    Args:
    model: Trained model object.
    model_type (str): Type of the model.
    evaluation_metric_value: Value of the evaluation metric.

    Returns:
    None
    """
    # Save model
    model_name = f"{model_type.lower()}_model.sav"
    joblib.dump(model, f"static/{model_name}")

    # Define evaluation metric
    evaluation_metric = "RMSE" if model_type == "Regression" else "Accuracy"

    # Log model details
    log_entry = {
        "Model Name": model_name,
        "Model train date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Evaluation Metric": evaluation_metric,
        "Model Evaluation Metric Value": evaluation_metric_value
    }

    # Load existing log data
    try:
        with open("static/train_log.json", "r") as log_file:
            log_data = json.load(log_file)
    except FileNotFoundError:
        log_data = {}

    # Update log data
    if model_type in log_data:
        log_data[model_type].append(log_entry)
    else:
        log_data[model_type] = [log_entry]

    # Write updated log data to file
    with open("static/train_log.json", "w") as log_file:
        json.dump(log_data, log_file, indent=4)


if __name__ == "__main__":
    
    # Check and delete existing train_log.json if it exists
    check_and_delete_log()
    
    # Read dataset
    df = read_data()

    # Plot species-wise count
    plot_species_count(df)

    # Plot correlation matrix
    plot_correlation_matrix(df)

    # Split data for regression
    train_x_reg, test_x_reg, train_y_reg, test_y_reg = preprocess_and_split_regression(df)

    # Train regression model
    reg_model = train_regression_model(train_x_reg, train_y_reg)

    # Evaluate regression model
    reg_predictions = reg_model.predict(test_x_reg)
    rmse = mean_squared_error(test_y_reg, reg_predictions, squared=False)

    # Save regression model and log
    save_model_and_log(reg_model, "Regression", rmse)

    # Split data for classification
    train_x_clas, test_x_clas, train_y_clas, test_y_clas = preprocess_and_split_classification(df)

    # Train classification model
    clas_model = train_classification_model(train_x_clas, train_y_clas)

    # Evaluate classification model
    clas_predictions = clas_model.predict(test_x_clas)
    accuracy = accuracy_score(test_y_clas, clas_predictions)

    # Save classification model and log
    save_model_and_log(clas_model, "Classification", accuracy)
