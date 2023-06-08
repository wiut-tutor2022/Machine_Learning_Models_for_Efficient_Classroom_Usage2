'''
Multiple Linear Regression
- is there way to graphically represent other dependent attributes as well, like:
week, day, time_of_day, class_type, school, joint, degree, class_duration, enrollment, status?
- at the moment showing only 'faculty' VS (predicting) 'normalized attendance' on plot
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

dataset = pd.read_csv(r'westminster.csv')
feature_cols = ['Week', 'Reg', 'Campus_Lecture_attendance', 'Module_X_Lecture_attendance', 'Module_A_deadlines', 'Campus_Lecture_attendance',
                'Module_B_deadlines', 'Module_C_deadlines']
# need to select only relevant labels
# Select the relevant features


# Extract the features and target variable
X = dataset[feature_cols]
print(X)
y = dataset['Module_X_Lecture_attendance']


train_data = pd.read_csv('westminster.csv')
test_data = pd.read_csv('westminster2.csv')


X_train = train_data.drop('Module_X_Lecture_attendance', axis=1)
y_train = train_data['Module_X_Lecture_attendance']

X_test = test_data.drop('Module_X_Lecture_attendance', axis=1)
y_test = test_data['Module_X_Lecture_attendance']

mlr = LinearRegression()  # #Fitting the Multiple Linear Regression model
mlr.fit(X_train, y_train)

# #Prediction of test set

y_pred_mlr = mlr.predict(X_test)

print("Intercept: ", mlr.intercept_)
print("Coefficients:")
list(zip(X, mlr.coef_))

# Predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

# Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()

print(mlr_diff)

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(mlr.score(X, y) * 100))
print('Mean Square Error:', meanSqErr)

##errors are generated with plot. Plot is not needed.
plt.scatter(dataset['Week'], dataset['Module_X_Lecture_attendance'], color='red')
plt.scatter(dataset['Week'], dataset['Campus_Lecture_attendance'], color='blue')
#
plt.title('Week VS (predicting) Module_X_Lecture_attendance', fontsize=14)
plt.xlabel('Week', fontsize=14)
plt.ylabel('Module_X_Lecture_attendance', fontsize=14)
plt.title('Multiple Linear Regression: On Campus Attendance vs Actual attendance')
plt.grid(True)
plt.show()
