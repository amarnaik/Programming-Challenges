'''
Author : Amar Naik

Program Description : This is Random Forest Classifier program. It reads data from 'train.csv' and does the training
                      Post the training the predictions and probabilities are applied on 'test.csv'
                      Finally a submission file 'Submission_Final.csv" in the correct format is created 
 


'''

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np
import csv

dftrain = pd.read_csv('E:/train.csv')
dftrain['DetectedCamera'] = dftrain['DetectedCamera'].map({'Rear': 0, 'Front': 1, 'Left': 2, 'Right': 3})
dftrain['SignFacing (Target)'] = dftrain['SignFacing (Target)'].map({'Rear': 0, 'Front': 1, 'Left': 2, 'Right': 3})

dftest = pd.read_csv('E:/test.csv')
dftest['DetectedCamera'] = dftest['DetectedCamera'].map({'Rear': 0, 'Front': 1, 'Left': 2, 'Right': 3})

#below command prints top 5 rows
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dftrain['Id'] = le.fit_transform(dftrain['Id'])
dftest['Id'] = le.fit_transform(dftest['Id'])
print('Number of observations in the training data:', len(dftrain))
print('Number of observations in the test data:', len(dftest))
# Create a list of the column's names without SignFacing column
SignFacing = dftrain.columns[:6]
#SignFacing = dftrain.columns[:0]
print(SignFacing)
# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, random_state=0)
n = len(dftest)
# Train the classifier to take the training features and learn how they relate
# to the values in column 'SignFacing (Target)'
clf.fit(dftrain[SignFacing], dftrain['SignFacing (Target)'])
# Capture the prediction of each row
output1=clf.predict(dftest[SignFacing])
# Capture the predicted probabilities of each row 
output2=clf.predict_proba(dftest[SignFacing])

# Open the test file to read and insert predictions
test_file_object = csv.reader(open("E:/test.csv", "r"))
next(test_file_object, None)  # skip the headers
#Open the file to write 
open_file_object = csv.writer(open("E:/testR1.csv", "w"))
#headers of the final output file
headers = ['Rear', 'Front', 'Left', 'Right','SignFacing (Target)','Id','DetectedCamera','AngleOfSign','SignAspectRatio','SignWidth','SignHeight']
open_file_object.writerow(headers)      

z = 0

for row in test_file_object:
            row.insert(0, str(output1[z])) # Insert the prediction at the start of the row
            row.insert(0, float(output2[z][3])) # Insert the prediction at the start of the row
            row.insert(0, float(output2[z][2])) # Insert the prediction at the start of the row
            row.insert(0, float(output2[z][1])) # Insert the prediction at the start of the row
            row.insert(0, float(output2[z][0])) # Insert the prediction at the start of the row
            open_file_object.writerow(row) # Write the row to the file
            z += 1
            if n == z :
                 break

#CREATE FINAL FILE
            
print("starting creation of final file")

df = pd.read_csv('E:/testR1.csv')

# select desired columns
df = df[['Id', 'Front','Left','Rear','Right']]
df.to_csv('E:/Submission_Final.csv', sep=',', index=False)
print('Observations predicted:', len(df))


#-----------------END---------------#