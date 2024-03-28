# -*- coding: utf-8 -*-

from flask import Flask, render_template, url_for ,request , redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from GATNN_function import *
import os 
# create empty model with the hyperparameter 
nCV = 10

my_hyper={'hidden_channels1': 112, 
 'hidden_channels2': 112,
 'heads1': 10, 
 'heads2': 10,
 'optimizer_type': 2, 
 'dropout_rateA': 0.1875877723990742,
 'dropout_rateB': 0.16097942424559475, 
 'dropout_rateC': 0.13247000729523636, 
 'learning_rate': 0.0001262534912805999, 
 'dense_layer1': 54,
 'decay': 0.0001409381594676602}

hidden_channels1= my_hyper['hidden_channels1']
hidden_channels2= my_hyper['hidden_channels2']
num_node_features =79
heads1=my_hyper['heads1']
heads2=my_hyper['heads2']
dropout_rateA=my_hyper['dropout_rateA']
dropout_rateB=my_hyper['dropout_rateB']
dropout_rateC=my_hyper['dropout_rateC']
dense_layer1=my_hyper['dense_layer1']

list_trained_model =[]
for i in range(10):
    loaded_model = modelA1(hidden_channels1,hidden_channels2, num_node_features,heads1,heads2,dropout_rateA,dropout_rateB,dropout_rateC,dense_layer1) 
    loaded_model.load_state_dict(torch.load("0.78915model_GNN"+ str(i)+ ".pth"))
    list_trained_model.append(loaded_model)


app = Flask(__name__)

picFolder =os.path.join('static','pics')
app.config['UPLOAD_FOLDER']= picFolder
#app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'

#db = SQLAlchemy(app)


  
    
@app.route('/', methods= ['POST','GET'])

def index():
    pred_result =""
    smiles_input = ""
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],"blokdiagram2.png")
    if request.method =='POST':
        smiles_input = request.form['content']
        
        pred_result = smiles_to_tuberc(smiles_input)
        return render_template('index.html',smiles_input=smiles_input, pred_result=pred_result, user_image=pic1)
        #request.form['result']=pred_result    
    else:
        return render_template('index.html',smiles_input=smiles_input,pred_result=pred_result, user_image=pic1)


    
if __name__=="__main__":
    app.run(debug=True)