from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Renders the form page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        features = pd.DataFrame([[
            float(request.form["loan_amnt"]),
            float(request.form["int_rate"]),
            float(request.form["annual_inc"]),
            float(request.form["dti"]),
            float(request.form["fico_range_high"])
        ]], columns=["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high"])

        # Scale the input data
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        # Convert prediction to human-readable text
        risk_level = "High Risk 🚨" if prediction[0] == 1 else "Low Risk ✅"

        return render_template("result.html", prediction=risk_level, features=features.to_dict(orient="records")[0])

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
