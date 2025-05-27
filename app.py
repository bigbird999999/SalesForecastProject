from flask import Flask, request, render_template
import joblib
import numpy as np
from xgboost import XGBRegressor

app = Flask(__name__)
model=XGBRegressor()
model.load_model("model/model.json")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    shop_id = int(request.form["shop_id"])
    item_id = int(request.form["item_id"])
    date_block_num = int(request.form["date_block_num"])
    lag1 = float(request.form["lag1"])
    lag2 = float(request.form["lag2"])
    lag3 = float(request.form["lag3"])
    rolling = np.mean([lag1, lag2, lag3])

    features = np.array([[date_block_num, shop_id, item_id, lag1, lag2, lag3, rolling]])
    prediction = model.predict(features)
    return render_template("index.html", prediction=round(float(prediction), 2))

if __name__=='__main__':
    app.run()