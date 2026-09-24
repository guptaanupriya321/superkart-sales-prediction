
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")
model = joblib.load("superkart_model.joblib")

FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

@superkart_api.get("/")
def home():
    return jsonify({"message": "Welcome to the SuperKart System", "status": "ok"})

@superkart_api.post("/v1/predict")
def predict_sales():
    payload = request.get_json(silent=True) or {}
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    try:
        input_data = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])
        prediction = float(model.predict(input_data)[0])
        return jsonify({"Sales": round(prediction, 2)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required under form field 'file'"}), 400

    try:
        input_data = pd.read_csv(request.files["file"])
        missing = [feature for feature in FEATURES if feature not in input_data.columns]
        if missing:
            return jsonify({"error": "Missing required columns", "missing": missing}), 400

        predictions = model.predict(input_data[FEATURES])
        output = input_data.copy()
        output["Predicted_Product_Store_Sales_Total"] = predictions.round(2)
        return jsonify(output.to_dict(orient="records"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
