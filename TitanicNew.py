# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt

# import seaborn by "pip install seaborn" 
import seaborn as sns
sns.set_style('whitegrid')

# import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Get titanic & test csv files as a DataFrame
# dtype={"Age": np.float64} tells python to treat Age column as float64 from the np library
titanic_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

# preview the data
print titanic_df.head()

# print details
titanic_df.info()
print("----------------------------")
test_df.info()

# drop the ticket columns in each, it is just a ticket number so is unlikely to be useful in analysis and prediction
titanic_df = titanic_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

# TITLE
# Create new title column and get title from passenger name
titanic_df['Title'] = titanic_df['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])
test_df['Title'] = test_df['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])

""" Taking out surname - do I actually want to use it for something?
titanic_df['Surname'] = titanic_df['Name'].apply(lambda x: x.split(',')[0].strip())
test_df['Surname'] = test_df['Name'].apply(lambda x: x.split(',')[0].strip())
"""
# Print first five values and print values and value counts for new column Title
print titanic_df.head()
print titanic_df['Title'].value_counts()
print test_df.head()
print test_df['Title'].value_counts()

# In titanic_df change Mlle and Ms to Miss, Mme to Mrs. Changed Col, Major and Capt to Military Title and all others to Rare Title
titanic_df['Title'] = np.where(titanic_df['Title'].isin(['Mlle', 'Ms']), 'Miss', titanic_df['Title'])
titanic_df['Title'] = np.where(titanic_df['Title'].isin(['Mme']), 'Mrs', titanic_df['Title'])
titanic_df['Title'] = np.where(titanic_df['Title'].isin(['Major', 'Col', 'Capt']), 'Military Title', titanic_df['Title'])
titanic_df['Title'] = np.where(titanic_df['Title'].isin(['Jonkheer', 'Lady', 'Don', 'Dona', 'the Countess', 'Sir']), 'Rare Title', titanic_df['Title'])
"""
# Plot Title vs Survived
title_perc = titanic_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()
sns.barplot(x = 'Title', y = 'Survived', data = title_perc)
fig = plt.gcf()
fig.savefig('titanic-Title.png')"""

# Make the same changes in test_df
test_df['Title'] = np.where(test_df['Title'].isin(['Mlle', 'Ms']), 'Miss', test_df['Title'])
test_df['Title'] = np.where(test_df['Title'].isin(['Mme']), 'Mrs', test_df['Title'])
test_df['Title'] = np.where(test_df['Title'].isin(['Major', 'Col', 'Capt']), 'Military Title', test_df['Title'])
test_df['Title'] = np.where(test_df['Title'].isin(['Jonkheer', 'Lady', 'Don', 'Dona', 'the Countess', 'Sir']), 'Rare Title', test_df['Title'])

# FAMILY
# Create a family size variable = number parents/children + number siblings/spouses + 1 for the passenger themselves
titanic_df['FamilySize'] =  titanic_df["Parch"] + titanic_df["SibSp"] + 1
test_df['FamilySize'] =  test_df["Parch"] + test_df["SibSp"] + 1
"""
# Plot the Family size showing number who survived and died
sns.countplot(x = 'FamilySize', hue= 'Survived', data = titanic_df)
fig = plt.gcf()
fig.savefig('titanic-FamilySize-Survival.png')"""

# Create function to be called on the FamilySize column and return singleton, small or large depending on the size of the family
def discrete_family_size(x):
    if x == 1:
        return 'Singleton'
    elif x > 4:
        return 'Large'
    else:
        return 'Small'

# Apply the function to create a new variable FamilySizeD with the discretized values 
titanic_df['FamilySizeD'] = titanic_df['FamilySize'].apply(discrete_family_size)        
test_df['FamilySizeD'] = test_df['FamilySize'].apply(discrete_family_size) 
sns.countplot(x = 'FamilySizeD', hue= 'Survived', data = titanic_df)
fig = plt.gcf()
fig.savefig('titanic-FamilySizeD.png')       

print titanic_df['FamilySizeD'].value_counts()

#MISSING VALUES

# EMBARKED
# Fill in the missing values for embarked using the value found as most likely by Megan Risdal's R script 
# where she looked at the fare paid by the people whose Embarked value is missing and looked at the others who paid this
# fare and found that Embarked for the group of people who paid this fare was Cherbourg
print titanic_df["Embarked"]
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("C")









