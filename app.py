from flask import Flask, request, jsonify
import joblib
import pandas as pd
import itertools

app = Flask(__name__)

# Load the trained model and associated functions from the saved model file
model_data = joblib.load('trained_model.pkl')

# Extract model, preprocess, and postprocess functions from the model dictionary
model = model_data['model']
preprocess_fn = model_data['preprocess_fn']
postprocess_fn = model_data['postprocess_fn']

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions based on time and device names.

    Returns:
        JSON: Predicted activation data for the next 24 hours for each device.
    """
    try:
        # Get data from the request and convert it to DataFrame
        input_data = request.get_json()
        next_24_hours = pd.date_range(input_data.get('time'), periods=24, freq="H").ceil("H")
        device_names = sorted(input_data.get('device'))
        xproduct = list(itertools.product(next_24_hours, device_names))
        predictions = pd.DataFrame(xproduct, columns=["time", "device"])

        # Preprocess and make predictions
        input_data = preprocess_fn(predictions)
        predictions["activation_predicted"] = model.predict(input_data)

        # Post-process predictions
        result = postprocess_fn(predictions)

        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app on all available network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000)
