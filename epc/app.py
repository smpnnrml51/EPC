from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv("gold.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Price"] = df["Price"].str.replace(",", "").astype(int)

# Prepare features and target
X = df["Date"].apply(lambda x: int(x.timestamp())).values
y = df["Price"].values

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X.reshape(-1, 1), y)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        input_date_str = request.form["input_date"]
        try:
            input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
            input_date_int = int(input_date.timestamp())
            predicted_price = model.predict([[input_date_int]])[0]
        except ValueError:
            predicted_price = "Invalid date format"

    return render_template("index.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
