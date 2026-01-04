from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the FULL pipeline (preprocessing + model)
pipeline = pickle.load(open("car_price_pipeline.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":

        # Collect raw inputs EXACTLY as trained
        input_data = {
            "model": request.form["model"],
            "vehicle_age": int(request.form["vehicle_age"]),
            "km_driven": int(request.form["km_driven"]),
            "seller_type": request.form["seller_type"],
            "fuel_type": request.form["fuel_type"],
            "transmission_type": request.form["transmission_type"],
            "mileage": float(request.form["mileage"]),
            "engine": int(request.form["engine"]),
            "max_power": float(request.form["max_power"]),
            "seats": int(request.form["seats"])
        }

        # Convert to DataFrame (VERY IMPORTANT)
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = round(pipeline.predict(input_df)[0], 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

