from flask import Flask, render_template, request
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'cancer20.pkl')
model = joblib.load(model_path)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the user
    try:
        # Get the features (input values from the form)
        features = [
            float(request.form['mean_radius']),
            float(request.form['mean_texture']),
            float(request.form['mean_perimeter']),
            float(request.form['mean_area']),
            float(request.form['mean_smoothness']),
            # Add all the required features here as per your model
        ]
        
        # Convert features to a numpy array for prediction
        features = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features)
        
        # Map prediction to human-readable labels
        result = 'breast cancer detected' if prediction[0] == 1 else 'No breast cancer detected'
        
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)