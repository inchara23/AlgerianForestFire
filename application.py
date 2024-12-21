from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pickle


application=Flask(__name__)
app=application

## app should be able to interact with pickle files
ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scaler=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))   # this Temperature should match with name given in html file
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        ## result will be a list of value which consists only one value so take first value from that

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run()  ## maps to the local systems ip address