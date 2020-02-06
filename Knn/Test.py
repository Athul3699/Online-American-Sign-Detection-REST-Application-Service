import csv
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle
import json

filLoc=input("Enter the path of the JSON file")
with open(filLoc) as json_file:
    handMovt = json.load(json_file)
featureMatrix=[]
featureRow=[]
for cols in range(7,11):
    score=[]
    x=[]
    y=[]
    for rows in range(0,len(handMovt)):
        score.append(float(handMovt[rows]["keypoints"][cols]["score"]))
        x.append(float(handMovt[rows]["keypoints"][cols]["position"]["x"]))
        y.append(float(handMovt[rows]["keypoints"][cols]["position"]["y"]))
    
    featureRow.append(np.sqrt(np.mean(np.array(score)**2)))
    featureRow.append(max(score)-min(score))
    featureRow.append(np.mean(score))
    featureRow.append(np.std(score))
    featureRow.append(max(score))
    featureRow.append(skew(score))
    featureRow.append(np.var(score))
    featureRow.append(kurtosis(score))    

    featureRow.append(np.sqrt(np.mean(np.array(x)**2)))
    featureRow.append(max(x)-min(x))
    featureRow.append(np.mean(x))
    featureRow.append(np.std(x))
    featureRow.append(max(x))
    featureRow.append(skew(x))
    featureRow.append(np.var(x))
    featureRow.append(kurtosis(x))

    featureRow.append(np.sqrt(np.mean(np.array(y)**2)))
    featureRow.append(max(y)-min(y))
    featureRow.append(np.mean(y))
    featureRow.append(np.std(y))
    featureRow.append(max(y))
    featureRow.append(skew(y))
    featureRow.append(np.var(y))
    featureRow.append(kurtosis(y))


featureMatrix.append(featureRow)
classes = {0:'Communicate',1:'Really',2:'Fun',3:'Mother',4:'Hope',5:'Buy'}
feature_names=['RMS','Range','Mean','STD','Max','Skew''Variance','Kurtosis']
featureMatrix=np.array(featureMatrix)
loaded_model = pickle.load(open('knnpickle_file.sav', 'rb'))
x_new = featureMatrix
y_predict = loaded_model.predict(x_new)
labelNumber=1
label=[]
for k in y_predict:
    print(classes[k])
    labelNumber=labelNumber+1   
    label.append(k)





    