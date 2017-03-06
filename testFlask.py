#!flask/bin/python
from flask import Flask
from flask import request
from flask import render_template
from sklearn import svm
from sklearn.externals import joblib
import ast
import os.path

app = Flask(__name__)
app.run(host='0.0.0.0')

clf = svm.SVC(gamma=0.001, C=100.)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/input', methods=['POST'])
def inputData_Post():
    roadID = ast.literal_eval(request.form['dataIDText'])
    dir = ast.literal_eval(request.form['dataDirText'])
    day = ast.literal_eval(request.form['dataDayText'])
    time = ast.literal_eval(request.form['dataTimeText'])
    sample = [ [ roadID , dir , day , time ] ]
    if os.path.isfile('data.pkl'):
        sample = joblib.load('data.pkl') + sample
    joblib.dump(sample, 'data.pkl')

    tar = ast.literal_eval(request.form['dataTargetText'])
    target = [ tar ]
    if os.path.isfile('target.pkl'):
        target = joblib.load('target.pkl') + target
    joblib.dump(target, 'target.pkl')

    return "Model was fitted with data."

@app.route('/predict', methods=['POST'])
def predict_Post():
    sample = joblib.load('data.pkl')
    target = joblib.load('target.pkl')
    clf.fit(sample, target)
    roadID = ast.literal_eval(request.form['predictIDText'])
    dir = ast.literal_eval(request.form['predictDirText'])
    day = ast.literal_eval(request.form['predictDayText'])
    time = ast.literal_eval(request.form['predictTimeText'])
    evaluatedText = [ roadID, dir, day, time ]
    return str(clf.predict(evaluatedText))

if __name__ == '__main__':
    app.run(debug=True)