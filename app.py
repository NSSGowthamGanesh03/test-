#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = model.predict(to_predict_list)
        if int(result)== 1: 
            prediction ='Engine Load is more than 50'
        else: 
            prediction='Engine Load is less than 50'            
    
        return render_template("index.html", prediction_text= prediction) 


if __name__ == '__main__':
    app.debug = True
    app.run(use_reloader=False)
    


# In[ ]:




