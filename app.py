from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

with open('temperature_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_temperature():
    # Get data from the form
    try:
        relative_humidity = float(request.form.get('Relative_Humidity'))
        pressure = float(request.form.get('Pressure'))
    except (TypeError, ValueError):
        return render_template('index.html', error="Please enter valid numbers for both fields.")

    # Convert data to a format compatible with the model
    input_features = np.array([[relative_humidity, pressure]])
    predicted_temp = model.predict(input_features)[0]

    # Return the prediction to the interface
    return render_template('index.html', prediction=predicted_temp)

if __name__ == '__main__':
    app.run(debug=True)
