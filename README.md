Fish Prediction Project
This repository contains a project aimed at helping users identify fish species or estimate the weight of a fish based on available data.

Dataset
Fish.csv: This file contains the dataset used for training and testing the machine learning models. It includes information about various fish species, such as weight, length, height, and width.
Templates
Index.html: This HTML file serves as the main entry point for the application. It provides options for users to choose between fish classification and regression tasks.
Classification.html: HTML template for the fish species classification page. It allows users to input fish features and predicts the species.
Regression.html: HTML template for the fish weight regression page. It allows users to input fish features and predicts the weight.
Static
script.js: JavaScript file containing client-side scripts for the application. It handles user interactions and communicates with the server.
styles.css: CSS file for styling the HTML pages to improve the user interface.
train_regression_model.sav: Trained regression model saved using joblib. This model predicts the weight of a fish based on its features.
train_classification_model.sav: Trained classification model saved using joblib. This model predicts the species of a fish based on its features.
StandardScaler.pkl: Serialized StandardScaler model for data preprocessing. It standardizes the numerical features of the dataset.
images/: Directory containing images of various fish species used for visualization on the web pages.
app.py
This Python file contains the Flask application code responsible for serving web pages, handling user requests, and making predictions based on machine learning models.

Procfile
A configuration file for Heroku deployment, specifying the command to run the web application.

training.py
Python script for training machine learning models using the provided dataset. It includes functions for data preprocessing, model training, evaluation, and saving trained models and logs.

train_log.json
A JSON file that logs details of the trained models, including model names, training dates, evaluation metrics, and evaluation metric values. This log is used to display model details on the web pages.
