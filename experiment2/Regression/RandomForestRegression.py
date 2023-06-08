
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
#dataset = pd.read_csv(r'../wiut.csv')
dataset = pd.read_csv(r'../westminster.csv')

#X = dataset.drop(['Week', 'Reg',], axis=1)  #

# Select the relevant features
feature_cols = ['Module_A', 'Module_B', 'Module_C',]

# Extract the features and target variable
X = dataset[feature_cols]
print(X)
y = dataset['Module_X_Seminar_attendance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = None)
#
# # Fit the random forest regression model
regr = RandomForestRegressor(n_estimators=100, max_depth=5)
regr.fit(X_train, y_train)
#
# # Generate predictions for the test set
y_pred = regr.predict(X_test)
#
# # Plot the predicted attendance against the actual attendance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual attendance')
plt.ylabel('Predicted attendance')
plt.title('Random Forest Regression: 3 features')
plt.show()
#
# # Compute the evaluation metrics
#
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(meanSqErr)
r_squared = regr.score(X, y)
#
#
print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R squared: {:.2f}'.format(r_squared * 100))
print('Mean Square Error:', meanSqErr)
#
