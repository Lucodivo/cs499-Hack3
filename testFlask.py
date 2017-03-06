#!flask/bin/python
from flask import Flask
from flask import request
from flask import render_template
from sklearn import svm
from sklearn.externals import joblib
import ast
import os.path

app = Flask(__name__)

clf = svm.SVC(gamma=0.001, C=100.)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/input')
def inputData():
    return render_template("inputData.html")

@app.route('/input', methods=['POST'])
def inputData_Post():
    data = ast.literal_eval(request.form['dataText'])
    if os.path.isfile('data.pkl'):
        data = joblib.load('data.pkl') + data
    joblib.dump(data, 'data.pkl')

    target = ast.literal_eval(request.form['targetText'])
    if os.path.isfile('target.pkl'):
        target = joblib.load('target.pkl') + target
    joblib.dump(target, 'target.pkl')

    clf.fit(data, target)
    return "Model was fitted with data."

@app.route('/predict')
def predict():
    return render_template("getPrediction.html")

@app.route('/predict', methods=['POST'])
def predict_Post():
    predictText = request.form['predictDataText']
    evaluatedText = ast.literal_eval(predictText)
    return str(clf.predict(evaluatedText))

if __name__ == '__main__':
    app.run(debug=True)