#!flask/bin/python
from flask import Flask
from flask import request
from flask import render_template
from sklearn import svm
import ast

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
    dataText = request.form['dataText']
    targetText = request.form['targetText']
    clf.fit(ast.literal_eval(dataText), ast.literal_eval(targetText))
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