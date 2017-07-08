'''
Author : Amar Naik

Program Description : This is Random Forest Classifier program. It reads data from 'train_data.json' and does the training
                      Post the training the predictions and probabilities are applied on 'test_data.json'
                      Finally a submission file 'Submission_Final.csv" in the correct format is created 
 


'''
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
#import numpy as np
import json
import csv

file_name_1 = "E:/Problem2/train_data.json"

with open(file_name_1, 'r') as jsonfile1:
    data_dict_1 = json.load(jsonfile1)
    
file_name_2 = "E:/Problem2/test_data.json"
with open(file_name_2, 'r') as jsonfile2:
    data_dict_2 = json.load(jsonfile2)

dftrain = pd.DataFrame.from_dict(data_dict_1, orient='index')
#train.reset_index(level=0, inplace=True)
dftrain.rename(columns = {'index':'ID'},inplace=True)
dftrain['segment'] = dftrain['segment'].map({'pos': 0, 'neg': 1})
print(dftrain.shape)
print(dftrain.columns[3])
#print(dftrain["segment"])

dftest = pd.DataFrame.from_dict(data_dict_2, orient='index')
#test.reset_index(level=0, inplace=True)
dftest.rename(columns = {'index':'ID'},inplace=True)
print(dftest.shape)
print(dftest.columns)
#print(test.head)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dftrain['titles'] = le.fit_transform(dftrain['titles'])
dftest['titles'] = le.fit_transform(dftest['titles'])
dftrain['genres'] = le.fit_transform(dftrain['genres'])
dftest['genres'] = le.fit_transform(dftest['genres'])
dftrain['cities'] = le.fit_transform(dftrain['cities'])
dftest['cities'] = le.fit_transform(dftest['cities'])
dftrain['tod'] = le.fit_transform(dftrain['tod'])
dftest['tod'] = le.fit_transform(dftest['tod'])
dftrain['dow'] = le.fit_transform(dftrain['dow'])
dftest['dow'] = le.fit_transform(dftest['dow'])

print('Number of observations in the training data:', len(dftrain))
print('Number of observations in the test data:', len(dftest))

# Create a list of the column's names without segment column
SEGtarget = dftrain.columns[:3]
y = pd.factorize(dftrain['segment'])[0]
#print(y)

#SignFacing = dftrain.columns[:0]
print(SEGtarget)
# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, random_state=0)
n = len(dftest)
# Train the classifier to take the training features and learn how they relate
# to the values in column 'segment'
print ("Train started")

clf.fit(dftrain[SEGtarget], y)
# Capture the prediction of each row
print ("Prediction started")
output1=clf.predict(dftest[SEGtarget])
# Capture the predicted probabilities of each row 
print ("probability")
output2=clf.predict_proba(dftest[SEGtarget])

#Open the file to write 
#open_file_object = csv.writer(open("E:/Problem2/testRA3.csv", "w"))
headers=["segment1","segment", "ID"]
#open_file_object.writerow(headers)      
z = 0
print ("starting writing to output file")

with open('E:/Problem2/Submission_Intermed.csv', 'w') as f:
        #f.write(headers)      
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(headers)
        for row in data_dict_2:
            m3=float(output2[z][1])
            m4=float(output2[z][0])
            istrr=[]
            istrr.append(m3)
            istrr.append(m4)
            istrr.append(row)
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(istrr)
            z += 1
            if n == z :
                 break
#open_file_object.writerow(row)

print ("Total Records written :",z)

#CREATE FINAL FILE
            
print("starting creation of final file")
df = pd.read_csv('E:/Problem2/Submission_Intermed.csv', sep=',')
#df.columns=["segment1","segment", "ID"]
# select desired columns
df = df[['ID', 'segment']]
df.to_csv('E:/Problem2/Submission_Final.csv', sep=',', index=False,header=True)
print ("END:Total Records written :",z)
exit


#-----------------END---------------#
