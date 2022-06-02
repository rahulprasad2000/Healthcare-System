# Liver

# Importing the libraries
import pandas as pd
import numpy as np

# Importing Dataset
dataset = pd.read_csv('liver.csv')
dataset = dataset.dropna(axis = 0)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,1] = labelencoder_x.fit_transform(X[:,1])

# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=0)
classifier.fit(X, y)

# Pickling the model
import pickle
file = open('liver.pkl','wb')
pickle.dump(classifier,file)
