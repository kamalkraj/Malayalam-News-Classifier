from flask import Flask,jsonify,request
from flask_cors import CORS
from predict import classify


app = Flask(__name__)
CORS(app)

@app.route("/classify",methods = ['POST'])
def _predict():
    text = request.json["text"]
    labels = classify(text)
    return jsonify(labels)

if __name__ == "__main__":
    app.run('0.0.0.0',port=5000)