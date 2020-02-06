import os
from glob import glob
import csv
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.neural_network import MLPClassifier
PATH = "/Users/athulpramod/Desktop/Mobile_Computing/classifier/CSV"
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

np.set_printoptions(suppress=True)
featureMatrix=[]
label=[]

for csvs in range(0,len(all_csv_files)):
    with open(all_csv_files[csvs], newline='') as csvfile:
        handMovt = list(csv.reader(csvfile)) 
    featureRow=[]
    if(csvs<72):
        label.append(0)
    else:
        if(csvs>=72 and csvs<143):
            label.append(1)
        else:
            if(csvs>=143 and csvs<214):
                label.append(2)
            else:
                if(csvs>=214 and csvs<280):
                    label.append(3)
                else:
                    if(csvs>=280 and csvs<349):
                        label.append(4)
                    else:
                        label.append(5)
                                
    for cols in range(23,35):
        y=[]
        for rows in range(1,len(handMovt)):
            y.append(float(handMovt[rows][cols]))
        featureRow.append(np.sqrt(np.mean(np.array(y)**2)))
        featureRow.append(max(y)-min(y))
        featureRow.append(np.mean(y))
        featureRow.append(np.std(y))
        featureRow.append(max(y))
        featureRow.append(skew(y))
        featureRow.append(np.var(y))
        featureRow.append(kurtosis(y))
    featureMatrix.append(featureRow)

feature_names=['RMS','Range','Mean','STD','Max','Skew','Variance','Kurtosis']
featureMatrix=np.array(featureMatrix)
label=np.array(label)

X = featureMatrix
y = label
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=2)


mlp = MLPClassifier(hidden_layer_sizes=(90,90,90), max_iter=1000)
mlp.fit(X,y)
mlpPickle = open('mlppickle_file.sav', 'wb') 
pickle.dump(mlp, mlpPickle)
x_new = featureMatrix
y_predict = mlp.predict(X_test)
score=metrics.accuracy_score(y_test,y_predict)


print("The accuracy is:",score*100)
