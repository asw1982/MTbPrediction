# GATNM : Graph with Attention Neural Network Model for Mycobacterial Cell Wall Permeability of Drugs and Drug-like Compounds
We developed GATNM as a proposed model for the mycobacterial tuberculosis cell wall permeability of drugs. This model is based on the graph attention neural network model complemented by the fully connected neural network as a classification layer. Before we got GATNM, we had carried out some experiments using many different models. All the codes here arranged sequentially in these folders, were used when our experiments were conducted. Three experiments that we ran consisted of machine learning, deep learning, and graph with attention neural network experiments.  

<img src="https://github.com/asw1982/MTbPrediction/assets/99703772/f947662d-43ab-451c-b7fe-0fe346ed3a74" width="800" />

# Manual 
We coded the model using the Jupyter Notebook platform. All the models were created based on the Pytorch and some data features were generated by using RDkit library packages. 
The environment was set by Anaconda and it can be seen in the **"environment.txt"** file. 
The dataset is also provided in **"all_dataset_mtbpen5371.csv"**

# 2. Data_preprocessing
In this folder, we made data preprocessing to repair, select, reduce, and find the most important data feature.
the file **"new_preprocessing_data.ipynb"** will generate the finally clean data. 
all the data would be saved as graph-structured data, reduced fingerprint data, and reduced descriptor data. 

# 3.1 machine_learning_experiment
To run 15 different machine-learning methods, we used Ilearnplus as a tool that can generate all the metrics in every model easily. 
the Ilearnplus library package is used when running this tool. the prepared dataset with the new format for Ilearnplus are provided in 
 - **"training_descriptor.csv"**
 - **"training_fingerprint.csv"**
 - **"test_descriptor.csv"**
 - **"test_fingerprint.csv"**

after running the code **"ilearnplus_app.ipynb"** then the application will be opened and ready to use. 
the performance result generated by Ilearnplus can be saved in .tsv file. 
<img src="https://github.com/asw1982/MTbPrediction/assets/99703772/8e660fea-6680-4ab8-b347-7319263d942c" width="800" />

# 3.2 single_input_fingerp_experiment 
This experiment was running when we applied a fully connected neural network using fingerprint input. 
There are two experiments in this folder. The first is used for getting the best model and the best hyperparameter by Optuna in code **"training_fingerp_FCNN.ipynb"**
after getting the best model then we run and record the performance of the best model in code **"single_FCNN_fingerp.ipynb"**

![image](https://github.com/asw1982/MTbPrediction/assets/99703772/eeaef180-089c-4e4f-b705-3f9d6b34b047)
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/a9774b4c-b79f-4c1d-960e-dfcb014110e0)  
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/4db29057-cac4-4887-a3ef-6c3aa33e1991)

# 3.3 single_input_desc_experiment
Similar to the previous experiment, we applied a fully connected neural network but using descriptor input. 
These files can be run to get all the performance results.
  - **"training_desc_FCNN.ipynb"**
  - **"single_FCNN_desc.ipynb"**
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/cf1a4b17-7d2c-412d-aad2-b8813b0cc189)
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/d89632d8-489c-4418-8b32-1403f8d04d98)
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/f0d8aa0c-dee2-41d2-ba72-137535260829)

# 3.4 single_input_graph_experiment
  - **"model_only_graph.ipynb"**
  - **"single_GATNN.ipynb"**
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/0d2615c5-4844-4db2-ba55-de268cc54934)
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/1cb433fd-da4a-439a-a84a-8aa437a4a462)
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/545945da-d26b-4173-ab9b-b39ab8396f1b)

# 3.5 ensemble_experiment
![ensemble_model](https://github.com/asw1982/MTbPrediction/assets/99703772/db73c0d3-f743-4282-8ab7-ae650903847e)
 - **"combine_pretrained_model_1.ipynb"**
 - **"perf_combined_model.ipynb"**
![image](https://github.com/asw1982/MTbPrediction/assets/99703772/4d7b078e-12ce-4c9c-983e-3be1ba602286)

# 4. web_GATNN_apps 


# email : 
agung.unitel@gmail.com 
