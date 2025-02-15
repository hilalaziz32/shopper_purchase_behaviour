from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("shopping_model.pkl")

# Initialize Flask app
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input from form
            admin = float(request.form["Administrative"])
            info = float(request.form["Informational"])
            product = float(request.form["ProductRelated"])
            exit_rate = float(request.form["ExitRates"])
            page_value = float(request.form["PageValues"])

            # Create input array
            user_input = np.array([[admin, info, product, exit_rate, page_value]])

            # Predict using the trained model
            prediction = model.predict(user_input)[0]

            # Convert prediction to meaningful output
            result = "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"

            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", result=f"Error: {e}")

    return render_template("index.html", result=None)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=False)

