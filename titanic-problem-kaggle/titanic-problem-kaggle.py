### IMPORTANT DISCLAIMER

# I was able to write this with significant help from https://careerhigh.in/blog/23/.
# This is *not* original code.

######################################

# IMPORTING LIBRARIES AND DATA

# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Loading the training file
titanic = pd.read_csv('titanic-problem-kaggle/data/train.csv')

# EXPLORING THE DATA

# sns.set_style('whitegrid')

# sns.countplot(x='Survived', hue='Sex', data= titanic,palette='RdBu_r')

# sns.countplot(x='Survived', hue='Pclass', data= titanic, palette='rainbow')

# sns.distplot(titanic['Age'].dropna(),color='darkred',bins=30)

# titanic['Fare'].hist(color='green',bins=40,figsize=(8,4))

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=titanic,palette='winter')

# plt.show()

# CLEANING AND TRANSFORMING THE DATA

# Filling empty age values with class-specific means

def impute_age(columns):
   Age = columns[0]
   Pclass = columns[1]
   if pd.isnull(Age):
       if Pclass == 1:
           return 37
       elif Pclass == 2:
           return 29
       else:
           return 24
   else:
       return Age

titanic['Age'] = titanic[['Age', 'Pclass']].apply(impute_age, axis = 1)

# Booleanise whether a row has a cabin number or not

def impute_cabin(column):
   Cabin = column[0]
   if type(Cabin) == str:
       return 1
   else:
       return 0

titanic['Cabin'] = titanic[['Cabin']].apply(impute_cabin, axis = 1)

# turn categorical features into dummy variables

# 'Let's work on a copy of our present dataset for further operations'
dataset = titanic

sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)
dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset = pd.concat([dataset,sex,embark],axis=1)

# PREPARE DATA FOR TRAINING AND TESTING

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Survived',axis=1),dataset['Survived'], test_size=0.25,random_state=101)

# LOGISTIC REGRESSION

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)

# 'Let’s evaluate our results on the X_test part of the dataset.'

print(accuracy_score(y_test, pred))

# QUESTIONS

# How can I adjust hyperparameters?
# Not sure what train_test_split actually does or why we're doing it
# How can I visually represent the regression?
# How can apply my new algorithm to the test data set?
# How can I submit this to Kaggle?


