from sys import stderr

from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
from pandas import array

app = Flask(__name__)

model = joblib.load('house.joblib')
# col=['stories_four','stories_one','stories_three','stories_two','lotsize','bedrooms','bathrms','driveway','recroom','fullbase','gashw','airco','garagepl','prefarea']

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final=[np.array(float_features, dtype=float)]
    prediction=model.predict(final)
    # output=round(prediction[0],2)

    return render_template('index.html', pred=prediction)



    # if(request.method=="POST"):
        # crim=float(request.form.get('CRIM'))
        # zn=float(request.form.get("ZN"))
        # indus=float(request.form.get("INDUS"))
        # chas=float(request.form.get("CHAS"))
        # nox=float(request.form.get("NOX"))
        # rm=float(request.form.get("RM"))
        # age=float(request.form.get("AGE"))
        # dis=float(request.form.get("DIS"))
        # rad=float(request.form.get("RAD"))
        # tax=float(request.form.get("TAX"))
        # ptratio=float(request.form.get("PTRATIO"))
        # b=float(request.form.get("B"))
        # lstat=float(request.form.get("LSTAT"))
        

    # features=array[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]

    

if __name__ == '__main__':
    app.run(debug=True)