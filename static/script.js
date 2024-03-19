// Function to fetch model details and populate the page

// Function to fetch and display model details
function getModelDetails(modelType) {
    fetch(`/get_model_training_details?modelType=${modelType}`)
        .then(response => response.json())
        .then(data => {
            if (data) {
                // Populate model details in the page
                document.getElementById('modelName').textContent = data['Model Name'];
                document.getElementById('modelTrainDate').textContent = data['Model train date'];
                document.getElementById('numRecords').textContent = data['Num Records Model Was Trained On'];
                document.getElementById('modelMetric').textContent = `${data['Model Evaluation Metric']}: ${data['Model Evaluation Metric Value']}`;
            } else {
                console.log(`No model details found for ${modelType}.`);
            }
        })
        .catch(error => console.error('Error fetching model details:', error));
}

// Function to predict species in classification
function predictSpecies() {
    // Get form data
    const formData = new FormData(document.getElementById('classificationForm'));

    // Convert form data to JSON
    const jsonData = {};
    formData.forEach((value, key) => {
        jsonData[key] = value;
    });

    // Make prediction request
    fetch('/predict_species', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction result
        document.getElementById('result').innerHTML = `<h2>Prediction Result</h2>
        <p>Identified Species: ${data.species}</p>
        <p>Probability: ${data.probability}</p>
        <img src="static/images/${data.image}" alt="${data.species}" class="fish-image" style="width: 6cm; height: 6cm;" title="${data.species}">
        `;
        // Update model training details for classification
        displayModelTrainingDetails('Classification');
    })
    .catch(error => console.error('Error predicting species:', error));
}

// Function to predict weight in regression
function predictWeight() {
    // Get form data
    const formData = new FormData(document.getElementById('regressionForm'));

    // Convert form data to JSON
    const jsonData = {};
    formData.forEach((value, key) => {
        jsonData[key] = value;
    });

    // Make prediction request
    fetch('/predict_weight', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction result
        document.getElementById('result').innerHTML = `<h2>Prediction Result</h2>
        <p>Predicted Weight: ${data.weight}</p>
        `;
        // Update model training details for regression
        displayModelTrainingDetails('Regression');
    })
    .catch(error => console.error('Error predicting weight:', error));
}


// Function to fetch and display model training details
function displayModelTrainingDetails(modelType) {
    getModelDetails(modelType); // Fetch model details based on model type

    // Display model training details
    // document.getElementById('modelTrainingDetails').innerHTML = `
       // <h2>Model Training Details</h2>
        // <p>Model Type: <span id="modelType"></span></p>
        // <p>Model Name: <span id="modelName"></span></p>
        // <p>Model Train Date: <span id="modelTrainDate"></span></p>
        // <p>Model Evaluation Metric: <span id="modelEvaluationMetric"></span></p>
        // <p>Model Evaluation Metric Value: <span id="modelEvaluationMetricValue"></span></p>
    //`;
}


// Function to update image based on selected species in regression
function updateImage() {
    var species = document.getElementById("species").value;
    var imgSrc = "";  // Add the appropriate image source for each species
    switch(species) {
        case "Bream":
            imgSrc = "static/images/Bream.jpg";
            break;
        case "Parkki":
            imgSrc = "static/images/Parkki.jpg";
            break;
        case "Perch":
            imgSrc = "static/images/Perch.jpg";
            break;
        case "Pike":
            imgSrc = "static/images/Pike.jpg";
            break;
        case "Roach":
            imgSrc = "static/images/Roach.jpg";
            break;
        case "Smelt":
            imgSrc = "static/images/Smelt.jpg";
            break;
        case "Whitefish":
            imgSrc = "static/images/Whitefish.jpg";
            break;
        default:
            imgSrc = "static/images/Bream.jpg"; // Set default image source to Bream if species not found
    }
    document.getElementById("speciesImage").src = imgSrc;
}
