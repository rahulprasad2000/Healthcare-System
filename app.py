# Healthcare App.py

from flask import Flask, render_template, request,Response,redirect, url_for,flash
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model
from skimage import transform
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import os
import glob
import re
import sys
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import pandas as pd
app = Flask(__name__)




###############################################################################
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
###############################################################################
   
    
###############################################################################
@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    model = pickle.load(open('diabetes.pkl','rb'))
    try:
        if request.method == 'POST':
            preg = float(request.form['preg'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['bp'])
            st = float(request.form['st'])
            ins = float(request.form['ins'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])
            result = model.predict([[preg,glucose,bp,st,ins,bmi,dpf,age]])
            if result[0] == 0:
               return render_template('diabetes.html',result='Congratulations! You do not have diabetes.')
            elif result[0] == 1:
               return render_template('diabetes.html',result='You have a very high chance of having Diabetes!')
            else:
               return render_template('home.html')
    except ValueError:
        return render_template('diabetes.html',result='Invalid Input')
        
    return render_template('diabetes.html')
###############################################################################
    

###############################################################################
@app.route('/heart',methods=['GET','POST'])
def heart():
    model = pickle.load(open('heart.pkl','rb'))
    try:
        if request.method == 'POST':
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])
            result = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,
                                           exang,oldpeak,slope,ca,thal]])
            if result[0] == 0:
                return render_template('heart.html',result='Congratulations! Your heart seems to be healthy!')
            elif result[0] == 1:
               return render_template('heart.html',result='Your heart is not in great shape!')
            else:
               return render_template('home.html')
    except ValueError:
        return render_template('heart.html',result='Invalid Input')
    
    return render_template('heart.html')    
###############################################################################
    

###############################################################################
@app.route('/liver',methods=['GET','POST'])
def liver():
    model = pickle.load(open('liver.pkl','rb'))
    try:
        if request.method == 'POST':
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            tb = float(request.form['tb'])
            db = float(request.form['db'])
            ap = float(request.form['ap'])
            alt = float(request.form['alt'])
            ast = float(request.form['ast'])
            tp = float(request.form['tp'])
            alb = float(request.form['alb'])
            ag = float(request.form['ag'])
            result = model.predict([[age,sex,tb,db,ap,alt,ast,tp,alb,ag]])
            if result[0] == 1:
                return render_template('liver.html',result='You have a problem in your liver')
            elif result[0] == 2:
                return render_template('liver.html',result='Congratulations! Your liver is fine.')
    except ValueError:
        return render_template('liver.html',result='Invalid Input')
    
    return render_template('liver.html')
###############################################################################
    

###############################################################################
MODEL_PATH = 'models/my_model.h5'

#Load your trained model
model = load_model(MODEL_PATH)      # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(50,50)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/malaria', methods=['GET', 'POST'])
def malFunc():
    try:
       if request.method == 'POST':
        # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            pred = model_predict(file_path, model)
            os.remove(file_path)#removes file from the server after prediction has been returned

            # Arrange the correct return according to the model. 
            # In this model 1 is Pneumonia and 0 is Normal.
            if pred[0] == 0:
               result= "Malaria Parasitized"
            else:  # Convert to string
               result="Normal"
            return result
    except ValueError:
        return render_template('malaria.html',result='Invalid Input')
    
    return render_template('malaria.html')
###############################################################################
model_path='models/pneumonia_model.h5'
model=load_model(model_path)

def model_predict(img_path,model):
    img_dims=224
    np_image=Image.open(img_path)
    np_image=np.array(np_image).astype('float32')/255
    np_image=transform.resize(np_image,(img_dims,img_dims,3))
    np_image=np.expand_dims(np_image,axis=0)
    preds=model.predict(np_image)
    return preds 

@app.route('/predict',methods=['GET','POST'])
def upload():
    try:
        if request.method=='POST':
            f=request.files['file']
            basepath=os.path.dirname(__file__)
            file_path=os.path.join(
            basepath,'uploads',secure_filename(f.filename))
            f.save(file_path)
            preds=model_predict(file_path,model)
            pred_class=preds[0][0]
            pred_class=int(np.round(pred_class))
            os.remove(file_path)
            if pred_class==1:
                result='This is PNEUMMONIA case'
                # return render_template('pneumonia.html',result='This is PNEUMMONIA case')
            else:
                result='This is a Normal Case'
                # return render_template('pneumonia.html',result='This is a Normal Case')
            return result
    except ValueError:
        return render_template('pneumonia.html',result='Invalid Input')

    return render_template('pneumonia.html')

#############################################################################
@app.route('/Hospital',methods=['GET','POST'])
def hospital():
    
    try:
        
        if request.method == 'POST':
            name = request.form['name']
            hospital_df = pd.read_csv('hospitalList.csv')
            dataframe = pd.DataFrame(hospital_df)
            dataframe['City']=dataframe['City'].str.lower()
            dataframe.drop('SL NO',axis=1,inplace=True)
            result_df=dataframe[dataframe['City']==name]
            
            
            return render_template('hospital.html',tables=[result_df.to_html(classes='hospital')])
            
    except ValueError:
        
        return render_template('hospital.html',result='Invalid Input')
    
    return render_template('hospital.html')
###############################################################################
@app.route('/details',methods=['GET'])
def details():
    return render_template('details.html')
###############################################################################
df = pd.read_csv('hospitalList.csv')
df.to_csv('hospitalList.csv', index=None)

@app.route('/table')
def table():
   data = pd.read_csv('hospitalList.csv')
   return render_template('table.html', tables=[data.to_html()], titles=[''])
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)