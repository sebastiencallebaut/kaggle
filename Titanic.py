# Titanic competition

# Work is inspired from
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
# https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic
# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

# 0. Import relevant packages for project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. Question and problem definition

# Competition goal is to predict, for each passenger, whether he will survive the sinking
# Personal goal is to create a simple model in few steps that delivers a benchmark against which we can fine tune our next models

# 2. Import the data (the data was already placed in the local directory

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combined_df = pd.concat([train_df, test_df])

# 3. Analyse the data

# First look at the data frames
print(train_df, combined_df)

# Check the column names
print(train_df.columns.values.tolist())

# Get the basic information about our data - From here we can see that Age, Fare, Cabin and Embarked have missing values
print(combined_df.info())

# Get basic stats about numerical data - From there we can see the on average young age, low fares paid, under 50% survivorship and mostly men customers
print(combined_df.describe(include='all'))

"""
# Plot some graphs to review basic insights - From this graph we can see that most perished, certainly in 3rd class
# Kids and women were safer
for element in ["Age", "Sex", "Embarked", "Fare"]:
    sns.catplot(x="Survived", y=element, col="Pclass", kind ="violin", data=combined_df);
plt.show()

# Plot correlations between numerical variables - No strong correlation between features. No risk of multicolinearity.
cor_metrics = combined_df[["Age", "Fare", "SibSp", "Parch"]].corr(method = 'pearson')
sns.heatmap(cor_metrics, annot=True, linewidths=.5, cmap="YlGnBu")
plt.show()
"""

# Missing values

# Fare - Only one missing value, we compose it based on the mean fare from people with same Pclass and port
print(combined_df[combined_df["Fare"].isna()])
combined_df["Fare"] = combined_df["Fare"].fillna(
    (combined_df.groupby(['Pclass', 'Embarked'])['Fare'].transform('mean')))
print(combined_df[combined_df["PassengerId"] == 1044])  # New fare is OK

# Cabin: we create a new column that is only the first letter of the cabin
combined_df["Cabin_letter"] = combined_df["Cabin"].str[0]
print(combined_df["Cabin_letter"].unique())
combined_df["Cabin_letter"] = combined_df["Cabin_letter"].fillna("Other")
combined_df = combined_df.drop("Cabin", axis=1)

# Embarked: Check the PassengerId of passengers with no port
print(combined_df[combined_df["Embarked"].isna()])
print(combined_df["Embarked"].value_counts())
combined_df["Embarked"] = combined_df["Embarked"].fillna("S")

# Age: replace it with the median age
median_age = np.nanmedian(combined_df["Age"])
mean_age = np.nanmean(combined_df["Age"])
combined_df["Age"] = combined_df["Age"].fillna(mean_age)

# Overview of all changes - no missing data anymore
print(combined_df.info())

# Features engineering for the name as we can extract the titles
combined_df["Title"] = combined_df["Name"].str.split(",", expand=True)[1].str.split(".", expand=True)[0].str[1:]
print(combined_df["Title"])

# One-hot encoding
print(combined_df["Ticket"])
combined_df = combined_df.drop(["Name", "Ticket"], axis=1)

combined_df = pd.get_dummies(combined_df)
print(combined_df.info())

# Split the data for test and training
train_df = combined_df[combined_df["Survived"].isin([1, 0])]
test_df = combined_df[combined_df["Survived"].isin([np.nan])]
print(train_df.info())
print(test_df.info())

# Train our model
y_train_df = train_df["Survived"]
x_train_df = train_df.drop(["Survived"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_df, test_size=0.3, random_state=12)

RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "min_samples_leaf": [1, 3, 10],
                 "bootstrap": [False],
                 "n_estimators": [100, 300],
                 "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=5, scoring="accuracy", n_jobs=4, verbose=1)

gsRFC.fit(x_train, y_train)

RFC_best = gsRFC.best_estimator_

"""
# Instantiate random forest
model = RandomForestClassifier(random_state=12)

# Instantiate params for GridSearch to iterate over
param_grid = {
    'n_estimators': [200,350, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8, 9],
    'criterion' :['gini', 'entropy']}

# Conduct the grid search
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)

# Get the best paramas
print(CV_rfc.best_params_)


# Use these estimators for our model
model1 = RandomForestClassifier(random_state=12, max_features='log2', n_estimators= 500, max_depth=8, criterion='gini')

# Fit the data
model1.fit(x_train, y_train)

# Make the prediction on the data
prediction = model1.predict(x_test)

# Get accuracy
print("Accuracy for Random Forest on test data: ",accuracy_score(y_test,prediction))

# Predictions for model
test_df = test_df.dropna(axis = 1)

# Make predictions on our data
op_rf = model1.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred.csv", index=False)

# Score of 0.78947 - Position 2727 out of 20574 on the 19th of August 2020
"""

# Ada boost

# Adaboost


# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                  "base_estimator__splitter": ["best", "random"],
                  "algorithm": ["SAMME", "SAMME.R"],
                  "n_estimators": [1, 2],
                  "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=5, scoring="accuracy", n_jobs=4, verbose=1)

gsadaDTC.fit(x_train, y_train)

ada_best = gsadaDTC.best_estimator_

"""
DTC = DecisionTreeClassifier()

model = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}


# Conduct the grid search
CV_ada = GridSearchCV(estimator=model, param_grid=ada_param_grid, cv= 5)
CV_ada.fit(x_train, y_train)

# Get the best paramas
print(CV_ada.best_params_)

# Use these estimators for our model
model2 = AdaBoostClassifier(DTC, random_state= 7 , algorithm = 'SAMME', base_estimator = 'gini', learning_rate = 0.0001, n_estimators = 1)

# Fit the data
model2.fit(x_train, y_train)

# Make the prediction on the data
prediction = model2.predict(x_test)

# Get accuracy
print("Accuracy for ADA on test data: ",accuracy_score(y_test,prediction))

# Predictions for model
test_df = test_df.dropna(axis = 1)

# Make predictions on our data
op_rf = model2.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred2.csv", index=False)
"""

# Gradient boosting


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss': ["deviance"],
                 'n_estimators': [100, 200, 300],
                 'learning_rate': [0.1, 0.05, 0.01],
                 'max_depth': [4, 8],
                 'min_samples_leaf': [100, 150],
                 'max_features': [0.3, 0.1]
                 }

gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=5, scoring="accuracy", n_jobs=4, verbose=1)

gsGBC.fit(x_train, y_train)

GBC_best = gsGBC.best_estimator_

"""
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }

# Conduct the grid search
CV_gb = GridSearchCV(estimator=GBC, param_grid=gb_param_grid, cv= 5)
CV_gb.fit(x_train, y_train)

# Get the best paramas
print(CV_gb.best_params_)


# Use these estimators for our model
model3 = GradientBoostingClassifier(random_state= 7 , learning_rate = 0.1, loss = 'deviance', max_depth = 4, max_features = 0.3, min_samples_leaf = 100, n_estimators = 200)

# Fit the data
model3.fit(x_train, y_train)

# Make the prediction on the data
prediction = model3.predict(x_test)

# Get accuracy
print("Accuracy for GB on test data: ",accuracy_score(y_test,prediction))

# Predictions for model
test_df = test_df.dropna(axis = 1)

# Make predictions on our data
op_rf = model3.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred3.csv", index=False)
"""

# SVC Classifier

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100, 200, 300, 1000]}

"""
# Conduct the grid search
CV_svc = GridSearchCV(estimator=SVMC, param_grid=svc_param_grid, cv= 5)
CV_svc.fit(x_train, y_train)

# Get the best paramas
print(CV_svc.best_params_)
"""
# Use these estimators for our model
model4 = SVC(random_state=7, C=1, gamma=0.001, kernel="rbf")

# Fit the data
model4.fit(x_train, y_train)

# Make the prediction on the data
prediction = model4.predict(x_test)

# Get accuracy
print("Accuracy for SVC on test data: ", accuracy_score(y_test, prediction))

# Predictions for model
test_df = test_df.dropna(axis=1)

# Make predictions on our data
op_rf = model4.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred4.csv", index=False)

# Extra trees


# ExtraTrees
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "min_samples_leaf": [1, 3, 10],
                 "bootstrap": [False],
                 "n_estimators": [100, 300],
                 "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=5, scoring="accuracy", n_jobs=4, verbose=1)

gsExtC.fit(x_train, y_train)

ExtC_best = gsExtC.best_estimator_

"""
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

# Conduct the grid search
CV_ext = GridSearchCV(estimator=ExtC, param_grid=ex_param_grid, cv= 5)
CV_ext.fit(x_train, y_train)

# Get the best paramas
print(CV_ext.best_params_)


# Use these estimators for our model
model5 = ExtraTreesClassifier(random_state=7,bootstrap = False, criterion= 'gini', max_depth = None,
             max_features = 3, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 300)

# Fit the data
model5.fit(x_train, y_train)

# Make the prediction on the data
prediction = model5.predict(x_test)

# Get accuracy
print("Accuracy for Extra tree on test data: ", accuracy_score(y_test, prediction))

# Predictions for model
test_df = test_df.dropna(axis=1)

# Make predictions on our data
op_rf = model5.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred5.csv", index=False)

"""

# Ensemble by voting

model6 = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best)
    , ('adac', ada_best), ('gbc', GBC_best)], voting='hard', n_jobs=4)

model6.fit(x_train, y_train)

# Make the prediction on the data
prediction = model6.predict(x_test)

# Get accuracy
print("Accuracy for Ensemble on test data: ", accuracy_score(y_test, prediction))

# Predictions for model
test_df = test_df.dropna(axis=1)

# Make predictions on our data
op_rf = model6.predict(test_df)

# Get the prediction
op = pd.DataFrame(test_df['PassengerId'])
op['Survived'] = op_rf

# Chnage type to int
op['Survived'] = op['Survived'].astype(int)

# Output the CSV file
op.to_csv("pred6.csv", index=False)

# Score of 0.80143 - Position 1036 out of 20114 on the 24th of August 2020