import numpy as np
import joblib
from flask import Flask, request, jsonify
app = Flask(__name__)


model = joblib.load("./mlp_model_latest.joblib")



@app.route("/test", methods=["GET"])
def test():
    try:
        response = {"error": "I am  here to test"}
        return response
    except Exception as e:
        response = {"error": str(e)}
        return response


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json 
        features = data["features"]
        # print("our check", type(features))
        
        givenArray = np.array([features])
        # print(givenArray)
        
        prediction = model.predict(givenArray)
        result = prediction.tolist()
        
        if result:
            identifiedClass = result[0]
            response = {"result": identifiedClass}
            return response
        else:
            response = {"error": "Empty result"}
            return response
    except Exception as e:
        response = {"error": str(e)}
        return response

if __name__ == "__main__":
    app.run(port=500)
   