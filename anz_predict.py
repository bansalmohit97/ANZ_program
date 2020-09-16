#ANZ DATA INSIGHTS

import pandas as pd

anz_dataset = pd.read_excel('ANZ synthesised transaction dataset-2.xlsx', index_col=None)
print(anz_dataset.shape)

anz_dataset

#HANDLE BPAY AND MERCHANT CODE COLUMNS
anz_dataset = anz_dataset.drop(['bpay_biller_code','merchant_code','merchant_id'], axis=1)

#Segment dataset into date and time

anz_dataset['transaction time'] = pd.to_datetime(anz_dataset['extraction']).dt.time

#Dummy encoding for gender column
anz_dataset['gender'] =pd.get_dummies(anz_dataset['gender'])

#Splitting the customer longitude and latitude
anz_dataset[['cust_long','cust_lat']] = anz_dataset['long_lat'].str.split(' -', expand=True)

#Drop extraction
anz_dataset=anz_dataset.drop('extraction', axis=1)

anz_dataset['date'] = pd.to_datetime(anz_dataset['date']).dt.date

anz_dataset['cust_lat']

#Dividing the dataset into authorized and posted since salary is just governed by posted transactions
anz_authorized = anz_dataset[anz_dataset['status']=='authorized']
anz_posted = anz_dataset[anz_dataset['status']=='posted']

#Dropping all the null values
anz_posted = anz_posted.dropna(axis = 1, how='any')

#Conditional dataset for just the employee's salary
anz_salary_dataset = anz_posted[(anz_posted['txn_description']=='PAY/SALARY') & (anz_posted['movement']=='credit')]

anz_salary_dataset['account']

#Annual Salary of a customer
salary_Account = pd.Series(anz_salary_dataset.groupby('account')['amount'].mean())
new_salaryDF = pd.DataFrame(anz_salary_dataset.groupby('account')['amount'].count())
new_salaryDF['salary_amount'] = salary_Account

#Creating conditions for weekly, fortnightly, monthy paid employers
for value in range(0, len(new_salaryDF['amount'])):
  if new_salaryDF['amount'][value] < 5:
    new_salaryDF['amount'][value] = 12    #monthly
  elif new_salaryDF['amount'][value] > 5 and value<12:
    new_salaryDF['amount'][value] = 26    #fortnightly
  else:
    new_salaryDF['amount'][value] = 52    #weekly

new_salaryDF['annual_salary'] = new_salaryDF['salary_amount'] * new_salaryDF['amount']
new_salaryDF = new_salaryDF.drop(['amount','salary_amount'], axis=1)

new_salaryDF

#Correlation
print(anz_salary_dataset)
anz_salary_dataset.corr()['amount']

#New dataset for the modelling
anz_posted_Main = anz_salary_dataset[['account','balance','gender','age','amount']]

#Normalizing the columns
anz_posted_Main[['balance','gender','age','amount']] = (anz_posted_Main[['balance','gender','age','amount']]-anz_posted_Main[['balance','gender','age','amount']].min())/(anz_posted_Main[['balance','gender','age','amount']].max()-anz_posted_Main[['balance','gender','age','amount']].min())
anz_posted_Main

#Correlation plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.scatter(anz_posted_Main['amount'],anz_posted_Main['balance'], label='Balance')
plt.scatter(anz_posted_Main['amount'],anz_posted_Main['gender'], label='Gender')
plt.scatter(anz_posted_Main['amount'],anz_posted_Main['age'], label='Age')
plt.legend()
plt.title('Correlation: Amount vs three features')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

#Outer Join

anz_FinalDF = anz_posted_Main.join(new_salaryDF, on = 'account', how='inner')

#Just gender and age of the employee would govern his annual income
anz_FinalDF = anz_FinalDF.drop(['balance','amount'], axis=1)

#Dropping duplicate rows
anz_FinalDF = anz_FinalDF.drop_duplicates()

#Normalizing the annual salary
anz_FinalDF['annual_salary'] = (anz_FinalDF['annual_salary']-anz_FinalDF['annual_salary'].min())/(anz_FinalDF['annual_salary'].max()-anz_FinalDF['annual_salary'].min())

anz_FinalDF

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(anz_FinalDF[['gender','age']],anz_FinalDF['annual_salary'], test_size = 0.3, shuffle = True)

#Linear Regression Model
lm_regress = linear_model.LinearRegression()
lm_regress.fit(X_train,Y_train)

pred_Annual_LR = lm_regress.predict(X_test)

#Performance for the model using MSE
MSE_LR = mean_squared_error(Y_test,pred_Annual_LR)
MSE_LR

"""---
**EXPLANATION:**
With a mean squared error of 0.034 which is very less, the model is very accurate. with this accuracy, it can certainly be used by ANZ for segmenting the 
customers as per their annual salaries. 

---
"""

#Decision tree model
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(max_depth = 4)
dt_model.fit(X_train,Y_train)

pred_Annual_DT = dt_model.predict(X_test)

#Performance for the model using MSE
MSE_DT = mean_squared_error(Y_test,pred_Annual_DT)
MSE_DT

"""---
**EXPLANATION:**
The mean squared error for the decision tree model is larger than that in the Linear Regression, which makes this model, less accurate than the former.

We can test the accuracy of this model by plotting the training and testing error for different depths of the tree.

---
"""