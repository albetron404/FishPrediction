from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import joblib
import pandas as pd


app = Flask(__name__)

# Define the path to the train log file
TRAIN_LOG_PATH = os.path.join(os.path.dirname(__file__), 'static/train_log.json')

# Load classification model
CLASSIFICATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'static/classification_model.sav')
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)

# Load regression model
REGRESSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'static/regression_model.sav')
regression_model = joblib.load(REGRESSION_MODEL_PATH)


@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/classification.html')
def classification():
    """Render the classification page with model details."""
    model_details = get_model_details("Classification")
    return render_template('classification.html', model_details=model_details)

@app.route('/regression.html')
def regression():
    """Render the regression page with model details."""
    model_details = get_model_details("Regression")
    return render_template('regression.html', model_details=model_details)

TRAIN_LOG_PATH = "static/train_log.json"

def get_model_details(model_type):
    """Return model details."""
    with open(TRAIN_LOG_PATH, 'r') as f:
        train_log_data = json.load(f)

    # Check if the model details are available in the JSON data
    if model_type in train_log_data:
        model_details = train_log_data[model_type][0]  # Assuming there's only one model of each type
    else:
        model_details = None

    return model_details

@app.route('/get_model_training_details', methods=['GET'])
def get_model_details_route():
    """Return model details."""
    model_type = request.args.get('modelType')
    model_details = get_model_details(model_type)
    return jsonify(model_details)






@app.route('/predict_species', methods=['POST'])
def predict_species():
    """Predict the species."""
    # Get features from request form
    features = request.json

    # Load the StandardScaler model
    scaler = joblib.load('static/standard_scaler_model.pkl')

    # Extract feature values in the correct order
    feature_values = [
        float(features['length1']), 
        float(features['length2']), 
        float(features['length3']), 
        float(features['height']), 
        float(features['width']),
        float(features['weight'])  # Include weight feature
    ]

    # Scale the features using the loaded scaler
    scaled_feature_values = scaler.transform([feature_values])

    # Get the predicted probabilities for each class
    probabilities = classification_model.predict_proba(scaled_feature_values)[0]

    # Get the class labels
    class_labels = classification_model.classes_

    # Create a DataFrame with class labels as index and probabilities as data
    df = pd.DataFrame(probabilities, index=class_labels, columns=['Probability'])

    # Find the class with the maximum probability
    max_prob_class = df.idxmax()[0]

    # Get the corresponding maximum probability
    max_prob = df.loc[max_prob_class]['Probability']

    # Determine the corresponding image file based on the predicted species
    image_file = max_prob_class.lower() + ".jpg"

    # Return the predicted class index, its corresponding probability, and the image file name
    return jsonify({'species': max_prob_class, 'probability': max_prob, 'image': image_file})




@app.route('/predict_weight', methods=['POST'])
def predict_weight():
    """Predict the weight."""
    # Get features from request form
    features = request.json

    # Extract numerical features
    numerical_features = [float(features[key]) for key in ['length1', 'length2', 'length3', 'height', 'width']]

    # Manually create one-hot encoding for 'Species'
    species_values = ['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish']
    species_encoded = [1 if features.get('species') == species else 0 for species in species_values]

    # Combine numerical and one-hot encoded features into a single list
    encoded_features = numerical_features + species_encoded

    # Call regression model method to predict weight
    weight = regression_model.predict([encoded_features])[0]

    # Return prediction results
    return jsonify({'weight': weight})

@app.route('/images/<path:filename>')
def images(filename):
    """Serve images."""
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    return send_from_directory(images_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
