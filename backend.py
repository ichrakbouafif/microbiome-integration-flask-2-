import joblib
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # Adjusted import
import pandas as pd

app = Flask(__name__)

# Load the models and scalers
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder_diag = joblib.load('label_encoder_diag.pkl')
label_encoder_stage = joblib.load('label_encoder_stage.pkl')

# Load feature names from the dataset
file_path = 'last_data1.csv'
data = pd.read_csv(file_path)
features = data.columns.drop(['DIAGNOSIS', 'STAGE', 'TUMOR_PERCENT'])
feature_names = features.tolist()[2:]  # Assuming the first two are to be ignored

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Parse input data from form
    age = request.form['age']
    fibro_vessel = request.form['fibro_vessel']
    sex = request.form['sex']
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    
    # Encode sex field
    sex_encoded = 1 if sex.lower() == 'male' else 0
    
    # Create input array
    input_values = [float(age), float(fibro_vessel), sex_encoded, float(feature1), float(feature2), float(feature3), float(feature4)]
    
    # Ensure the input data has exactly 7 features
    if len(input_values) != 7:
        return render_template('index.html', prediction='Error: Expected 7 input features.')
    
    # Map the provided 7 features to specific feature names (customize as needed)
    provided_features = {
        'AGE': input_values[0],
        'FIBROBLAST_AND_VESSEL_PERCENT': input_values[1],
        'SEX': input_values[2],
        'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Peptostreptococcaceae;g__Peptostreptococcus; s__anaerobius': input_values[3],
        'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Prevotellaceae; g__Prevotella; s__stercorea': input_values[4],
        'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales; f__Bacteroidaceae; g__Bacteroides': input_values[5],
        'k__Bacteria; p__Bacteroidetes; c__Bacteroidia; o__Bacteroidales': input_values[6]
    }

    # Fill missing features with 0 and ensure the order of features matches the training data
    sample_input = [provided_features.get(feature, 0) for feature in feature_names]
    
    # Convert to numpy array and ensure correct shape
    sample_input = np.array(sample_input).reshape(1, -1)
    
    # Normalize the sample input using the scaler
    sample_input_scaled = scaler.transform(sample_input)
    
    # Make predictions using the trained model
    predictions = model.predict(sample_input_scaled)
    
    # Debug print statements to understand the shape and content of predictions
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # Assuming the model outputs predictions for diagnosis and stage as separate outputs
    if len(predictions) == 2:  # If there are two outputs (e.g., diagnosis and stage)
        predicted_diag = np.argmax(predictions[0], axis=-1)
        predicted_stage = np.argmax(predictions[1], axis=-1)
    else:
        predicted_diag = np.argmax(predictions, axis=-1)
        predicted_stage = np.argmax(predictions, axis=-1)
    
    # Decode the predictions
    predicted_diag_label = label_encoder_diag.inverse_transform([predicted_diag])[0]
    predicted_stage_label = label_encoder_stage.inverse_transform([predicted_stage])[0] if predicted_stage != "N/A" else "N/A"
    
    return render_template('index.html', prediction=f'Predicted Diagnosis: {predicted_diag_label}, Predicted Stage: {predicted_stage_label}')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
