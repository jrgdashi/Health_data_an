import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.formula as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE

"""
Based on outside search I came to the conclusion that the only independent variables are the Glucose, Age.
Insulin depends on the amount of glucose in the body, which effects the blood pressure since it would thicken 
the blood and it would take the heart more strength to pump, insulin would also affect the skin-thickness.
BMI is affected from weight and height but in this case we have neither of those variables, but I can maybe
find a relation to it with patients that have a bad blood pressure, and from the relation between glucose level and
insulin level.
Pregnancies depend on age, but not really.
Diabetes Pedigree Function tells the prob. of someone getting diabetes based on family history.
"""

def scale(df,N):
    a = []
    for col in df.columns:
        a.append(df[col].groupby(df[col].index // N).mean())
    scaledf=pd.DataFrame(a)
    return scaledf.transpose()

df = pd.read_csv(r'C:\Users\jrgda\OneDrive\Desktop\new2.txt')
print(df.head())

# no null values are found
print(df.isnull().sum())

# there is no missing values, and all he variables are numbers
print(df.describe())

"""
Independent:
Glucose has a slight left skewed distribution.
The majority of the people's age looks to be in their 20's to early 30's so I would assume lower cases of diabetes.

Dependent:
Pregnancies have a right skewed distribution.
Blood Pressure had a bimodal distribution.
Skin thickness looks to have a right skewed distribution.
Insulin looks to have a right skewed distribution.
BMI look to be normally distributed.
DiabetesPedigreeFunction looks looks to have a right skewed distribution.
About 1/3 of the people have diabetes where as 2/3 don't.
"""
n=scale(df, 1)
sns.displot(n['Pregnancies'])
plt.show()
for col in df.columns:
    sns.displot(n[col])
    plt.show()
""" 
There is a high correlation between skin thickness and insulin with pregnancy.
"""
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# REGRESSION MODEL
"""
based in the correlation model I wanna see what effects the outcome
"""
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, df.columns == 'Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

# Checking values to make sure they are what I want, which the are.
print(y_train)
print(X_train)

#  x and y together created the train database
t_data = pd.concat([X_train, y_train], axis=1)

print(t_data.head())

model1= smf.ols('Outcome ~ DiabetesPedigreeFunction + Glucose', data= t_data).fit()

print(model1.summary())

coef = model1.params
rs = pd.Series([model1.rsquared], index=["R_squared"])
r1= coef.append(rs)
r1= pd.DataFrame(data=r1, columns=["Value"])
print(r1)

# K-fold

# this has low accuracy of just 23%
model1_score = cross_val_score(LinearRegression(), X_train[['DiabetesPedigreeFunction','Glucose']], y_train, cv=10)
print(model1_score)
print(np.mean(model1_score))

# RMSE
y_prediction= model1.predict(X_test[['DiabetesPedigreeFunction','Glucose']])
print(np.sqrt(metrics.mean_squared_error(y_test.T.squeeze(), y_prediction)))

# Model 2

X2 = df.loc[:, df.columns != 'Outcome']
y2 = df.loc[:, df.columns == 'Outcome']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=12)

# Checking values to make sure they are what I want, which the are.
print(y2_train)
print(X2_train)

#  x and y together created the train database
t_data2 = pd.concat([X2_train, y2_train], axis=1)

print(t_data2.head())

model2= smf.ols('Outcome ~ BloodPressure + DiabetesPedigreeFunction', data= t_data2).fit()

print(model2.summary())

coef2 = model2.params
rs2 = pd.Series([model2.rsquared], index=["R_squared"])
r2= coef2.append(rs2)
r2= pd.DataFrame(data=r2, columns=["Value"])
print(r2)

# K-fold

model2_score = cross_val_score(LinearRegression(), X2_train[['BloodPressure','DiabetesPedigreeFunction']], y2_train, cv=10)
print(model2_score)
print(np.mean(model2_score))

# RMSE
y2_prediction= model2.predict(X2_test[['BloodPressure','DiabetesPedigreeFunction']])
print(np.sqrt(metrics.mean_squared_error(y2_test.T.squeeze(), y2_prediction)))


sm = SMOTE(random_state=12)
X_res, y_res = sm.fit_resample(X_train , y_train)
X2_res, y2_res = sm.fit_resample(X2_train , y2_train)



X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, df.columns == 'Outcome']
X_res, X_test, y_res, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

# Checking values to make sure they are what I want, which the are.
print(y_res)
print(X_res)

#  x and y together created the train database
t_data = pd.concat([X_res, y_res], axis=1)

print(t_data.head())

model1= smf.ols('Outcome ~ DiabetesPedigreeFunction + Glucose', data= t_data).fit()

print(model1.summary())

coef = model1.params
rs = pd.Series([model1.rsquared], index=["R_squared"])
r1= coef.append(rs)
r1= pd.DataFrame(data=r1, columns=["Value"])
print(r1)

# K-fold

model1_score = cross_val_score(LinearRegression(), X_res[['DiabetesPedigreeFunction', 'Glucose']], y_res, cv=10)
print(model1_score)
print(np.mean(model1_score))

# RMSE
y_prediction= model1.predict(X_test[['DiabetesPedigreeFunction','Glucose']])
print(np.sqrt(metrics.mean_squared_error(y_test.T.squeeze(), y_prediction)))

# Model 2

X2 = df.loc[:, df.columns != 'Outcome']
y2 = df.loc[:, df.columns == 'Outcome']
X2_res, X2_test, y2_res, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=12)

# Checking values to make sure they are what I want, which the are.
print(y2_res)
print(X2_res)

#  x and y together created the train database
t_data2 = pd.concat([X2_res, y2_res], axis=1)

print(t_data2.head())

model2= smf.ols('Outcome ~ BloodPressure + DiabetesPedigreeFunction', data= t_data2).fit()

print(model2.summary())

coef2 = model2.params
rs2 = pd.Series([model2.rsquared], index=["R_squared"])
r2= coef2.append(rs2)
r2= pd.DataFrame(data=r2, columns=["Value"])
print(r2)

# K-fold

model2_score = cross_val_score(LinearRegression(), X2_res[['BloodPressure', 'DiabetesPedigreeFunction']], y2_res, cv=10)
print(model2_score)
print(np.mean(model2_score))

# RMSE
y2_prediction= model2.predict(X2_test[['BloodPressure','DiabetesPedigreeFunction']])
print(np.sqrt(metrics.mean_squared_error(y2_test.T.squeeze(), y2_prediction)))

"""after using the SMOTE the data got worse, I don't understand what I did wrong, since I'm getting values
that don't look right, in conclusion either the data it's not good or I did something wrong and I believe it's
probably the latter"""