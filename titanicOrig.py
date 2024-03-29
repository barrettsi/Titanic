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
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
# dtype={"Age": np.float64} tells python to treat Age column as float64 from the np library
titanic_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df    = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

# preview the data
print titanic_df.head()

# print details
titanic_df.info()
print("----------------------------")
test_df.info()

# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)
"""
# print details after drop 
print 'Dropping PassengerId, Name, Ticket'
titanic_df.info()
print("----------------------------")
print 'Dropping Name, Ticket'
test_df.info()
"""     
# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
# [The R version did a better job of this by checking where others who paid his fair embarked and found C
# need to be sure that when substituting values for NA we have made a logical choice]
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

#CREATES THE FIRST LINE PLOT WITH PERCENTAGE SURVIVED FOR EACH EMBARCATION POINT
# plot
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig = plt.gcf()
fig.savefig('titanic-3-axes.png')

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

# Is where they embarked from useful or is it because of how much they paid for their fare, booked a cabin
# therefore they were wealthy and more likely to get off the boat any way? This author thinks embarked is not important so he drops it
titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

fig.savefig('titanic_embark_survived.png')
"""
# print details after drop 
print 'Dropping Embarked after creating dummy variable columns C and Q'
titanic_df.info()
print("----------------------------")
print 'Dropping Embarked after creating dummy variable columns C and Q'
test_df.info()
"""
# Fare

# only for test_df, since there is a missing "Fare" values
# Filling in the missing value with the median Fare value
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int - WHY???
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

#SAVE THE PLOT OF FARE AS A HISTOGRAM - shows the frequency of survival
fig = plt.gcf()
fig.savefig('titanic-hist-fare.png')

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

#Plots the average fare of those who survived and and those who did not. The highest value paid by someone who died was 58
#(top of black line). Didn't give one group a better chance of surviving based on fare.
fig = plt.gcf()
fig.savefig('titanic-bar-ave-fare.png')

# AGE

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum() # take age column give me all those that are null and sum the amount(count the NANs)

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std) i.e. normal bell curve -sigma to sigma
# if done in a randomly enough way it will not change the underlying distribution. The random numbers themselves should be a normal
# distribution. You could just put all 177 missing to the mean but you could skew the data when you go to machine learning e.g. if you
# set all the ages to mean the ML algorithm may decide age is not important and leave it out when we know that women with children were
# sent to lifeboats first so age is important. Also if you just choose mean this will skew the standard deviation.
# if you use 0 to 81 (max and min) then you will end up with ages around the end values which will skew the distribution
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)

# plot original Age values
titanic_df['Age'].hist(bins=70, ax=axis1)
# test_df['Age'].hist(bins=70, ax=axis3)

fig = plt.gcf()
fig.savefig('titanic-hist-orig-age.png') # Best to not use this as there are 2 graphs side by side)
        
# plot new Age Values
titanic_df['Age'].hist(bins=70, ax=axis2)
# test_df['Age'].hist(bins=70, ax=axis4)

fig = plt.gcf()
fig.savefig('titanic-hist-fitted-age.png')
# After plotting we compare them and see that the histograms still look the same - the data has not been skewed

# .... continue with plot Age column
# Looking at average numbers of passengers who survived based on age
# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()

# If you were between 15 and 29 you you had a higher chance of dying, check out facet plots and what they mean
# This shows the normal distribution curve of the people who survived and who didn't survive. They are very similar but survived had bigger tails
# The whole green line adds up to 1 and the whole blue line adds up to one
fig = plt.gcf()
fig.savefig('titanic-facet-map.png')


# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
# There was a worse chance of surviving if you were between 14 and 47. Much better chance of surviving if you are over 48 or under 14 e.g.
# every child aged 0, 5, 12 and 13 survived. There may have been just one child aged 0 and they got on to a lifeboat first
fig = plt.gcf()
fig.savefig('titanic-bar-age.png')

# CABIN
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)
"""
# print details after drop 
print 'Dropping Cabin'
titanic_df.info()
print("----------------------------")
print 'Dropping Cabin'
test_df.info()
"""
# FAMILY
# redefine with dimension reduction
# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
# Megan Risdal looked at whether the size of the family matters and it does
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

"""
# print details after drop 
print 'Dropping SibSp and Parch after creating Family'
titanic_df.info()
print("----------------------------")
print 'Dropping SibSp and Parch after creating Family'
test_df.info()
"""

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

fig = plt.gcf() # this says give me the graphical frame as fig and now I can save fig. If I have fig = then I can just save the fig and don't need this line
fig.savefig('titanic-bar-family.png')

# SEX

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)
"""
# print details after drop 
print 'Dropping Sex after creating Person'
titanic_df.info()
print("----------------------------")
print 'Dropping Sex after creating Person'
test_df.info()
"""
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)
"""
# print details after dummies and drop
print 'Dropping Male after creating Child, Female, Male'
titanic_df.info()
print("----------------------------")
print 'Dropping Male after creating Child, Female, Male'
test_df.info()

"""
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

fig = plt.gcf()
fig.savefig('titanic-count-and-bar-sex.png')

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)
"""
# print details after drop
print 'Dropping Person'
titanic_df.info()
print("----------------------------")
print 'Dropping Person'
test_df.info()
"""
# PCLASS
# First class had a much higher level of survival
# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

fig = plt.gcf()
fig.savefig('titanic-factor-pclass.png')

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df = test_df.join(pclass_dummies_test)

"""
# print details after drop
print 'Dropping Class_3 and Pclass after creating Class_1 and Class_2'
titanic_df.info()
print("----------------------------")
print 'Dropping Class_3 and Pclass after creating Class_1 and Class_2'
test_df.info()
"""
titanic_df.info()
print titanic_df.head()
test_df.info()
print test_df.head()

# define training and testing sets
# X train and Y train are the same except for the removed survived column
X_train = titanic_df.drop("Survived",axis=1) # dropped the survived column
Y_train = titanic_df["Survived"] # have the answer
X_test  = test_df.drop("PassengerId",axis=1).copy()


# Logistic Regression
# checking the x train performance against y train and get 81%
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
lr_score = logreg.score(X_train, Y_train)

# Testing -log reg. This is the function taking in the data sets.
# def get_prediction_percentage(X_train, Y_train,)
#   logreg = LogisticRegression()
#   logreg.fit(X_train, Y_train)
#   Y_pred = logreg.predict(X_test)
#   return logreg.score(X_train, Y_train)
#
# def setUp(self):
#   self.titanic_df = 
#    
# def_test_prediction_percentage():
#   X_train = 
#   Y-train = 
#   for i in range(1, 100):
#       self.assertTrue(80 < get_prediction_percentage(X_train, Y_train,))
# might be happy with this provided it is always within the standard deviation or just over 80
# By doing the unittest I can test over and over again this model and see how many times it fails
# If it fails 3 times out of 20 then 85% confidence that this will give the required prediction certainty
# If 70 is the lowest value we get and the customer is happy with a value of 70 then we can say with 100% certainty that 
# our model will give at least 70

# Random Forests - 96%
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
rf_score = random_forest.score(X_train, Y_train)

# Support Vector Machines - 86%
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc_score = svc.score(X_train, Y_train)

# K Nearest Neighbours - 81%
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn_score = knn.score(X_train, Y_train)

# Gaussian Naive Bayes - 76%
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
gnb_score = gaussian.score(X_train, Y_train)

print 'Logistic regression score is {}. Random Forest score is {}. SVC score is {}. KNN score is {}. GNB score is {}'.format( lr_score, rf_score, svc_score, knn_score, gnb_score)
        

