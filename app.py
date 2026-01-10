# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open("house_price_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    price = None

    if request.method == "POST":
        data = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["GarageCars"]),
            float(request.form["YearBuilt"]),
            float(request.form["FullBath"]),
            float(request.form["BedroomAbvGr"]),
            float(request.form["LotArea"]),
        ]

        prediction = model.predict([data])[0]
        price = round(prediction, 2)

    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)
