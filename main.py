from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("svm_water_quality.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Ensure this file exists in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Map prediction to readable text
        result = "Potable (Safe to Drink)" if prediction[0] == 1 else "Not Potable (Unsafe to Drink)"

        return render_template("index.html", prediction_text=f"Water Quality: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
