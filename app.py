from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define feature names and ranges
FEATURES = ['age', 'sex', 'cp', 'trtbps', 'chol']
FEATURE_RANGES = {
    'age': (0, 120, float),
    'sex': (0, 1, int),
    'cp': (0, 3, int),
    'trtbps': (50, 200, float),
    'chol': (100, 600, float)
}

# In-memory prediction history
prediction_history = []

# Load or create dummy model and scaler
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    print("Successfully imported scikit-learn modules")
except ImportError as e:
    print(f"Fatal error: Failed to import scikit-learn modules: {str(e)}")
    raise ImportError(f"scikit-learn is required. Install with 'pip install scikit-learn'")

try:
    model = joblib.load('heart_attack_rf_model.pkl')
    scaler = joblib.load('heartattack.pkl')
    print("Loaded existing model and scaler")
except FileNotFoundError:
    print("Model/scaler files not found, creating dummy ones")
    X_dummy = np.random.rand(100, len(FEATURES))
    y_dummy = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    model.fit(scaler.transform(X_dummy), y_dummy)
    joblib.dump(model, 'heart_attack_rf_model.pkl')
    joblib.dump(scaler, 'heartattack.pkl')
    print("Created and saved dummy model and scaler")
except Exception as e:
    print(f"Error creating/loading model/scaler: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"Received request: {request.method} to {request.url}")
    if request.method == 'POST':
        print("POST data:", request.form)
        try:
            # Extract and validate input data
            data = request.form.to_dict()
            input_data = []
            print("Validating inputs...")
            for feature in FEATURES:
                if feature not in data:
                    print(f"Error: Missing {feature}")
                    return render_template('index.html', error=f"Missing {feature} in input.", history=prediction_history)
                try:
                    value = data[feature].strip()
                    if value == '':
                        print(f"Error: {feature} is empty")
                        return render_template('index.html', error=f"{feature} cannot be empty.", history=prediction_history)
                    value = float(value)
                    if FEATURE_RANGES[feature][2] == int:
                        if not value.is_integer():
                            print(f"Error: {feature} must be an integer")
                            return render_template('index.html', error=f"{feature} must be an integer.", history=prediction_history)
                        value = int(value)
                    min_val, max_val, _ = FEATURE_RANGES[feature]
                    if not (min_val <= value <= max_val):
                        print(f"Error: {feature} value {value} out of range [{min_val}, {max_val}]")
                        return render_template('index.html', error=f"{feature} must be between {min_val} and {max_val}.", history=prediction_history)
                    input_data.append(value)
                except ValueError as e:
                    print(f"Error: Invalid value for {feature}: {str(e)}")
                    return render_template('index.html', error=f"Invalid value for {feature}. Must be a number.", history=prediction_history)

            # Prepare input for model
            print("Preparing model input:", input_data)
            input_array = np.array([input_data])
            print("Input shape:", input_array.shape)
            try:
                input_scaled = scaler.transform(input_array)
                print("Scaled input:", input_scaled)
            except Exception as e:
                print(f"Error scaling input: {str(e)}")
                return render_template('index.html', error=f"Scaling error: {str(e)}", history=prediction_history)

            # Make prediction
            try:
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1] * 100
                print(f"Prediction: {prediction}, Probability: {probability}")
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return render_template('index.html', error=f"Prediction error: {str(e)}", history=prediction_history)

            # Prepare result with feature importance
            feature_importance = [
                {'feature': feature, 'importance': float(round(imp * 100, 2))}
                for feature, imp in zip(FEATURES, model.feature_importances_)
            ]
            print("Feature importance:", feature_importance)
            result = {
                'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                'probability': round(probability, 2),
                'input_data': {feature: value for feature, value in zip(FEATURES, input_data)},
                'feature_importance': feature_importance
            }
            print("Rendering result:", result)

            # history (limit to last 5 predictions)
            prediction_history.append({
                'inputs': result['input_data'],
                'prediction': result['prediction'],
                'probability': result['probability']
            })
            if len(prediction_history) > 5:
                prediction_history.pop(0)
            
            return render_template('index.html', result=result, history=prediction_history)
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return render_template('index.html', error=f"Server error: {str(e)}", history=prediction_history)
    
    print("Rendering index.html for GET request")
    return render_template('index.html', history=prediction_history)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except OSError as e:
        print(f"Failed to start server: {str(e)}")
        print("Ensure port 5000 is free. Run 'netstat -a -n -o | find \"5000\"' and 'taskkill /PID <pid> /F' to free it.")