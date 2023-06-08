#
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

# Load the dataset from a CSV file#


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
test_data = pd.read_csv('westminster3.csv')


X_train = train_data.drop('Module_X_Lecture_attendance', axis=1)
y_train = train_data['Module_X_Lecture_attendance']

X_test = test_data.drop('Module_X_Lecture_attendance', axis=1)
y_test = test_data['Module_X_Lecture_attendance']



# Fit the Multiple Linear Regression model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Prediction on the test set
y_pred_mlr = mlr.predict(X_test)

# Print model coefficients and intercept
print("Intercept:", mlr.intercept_)
print("Coefficients:")
print(list(zip(X.columns, mlr.coef_)))

# Calculate evaluation metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(meanSqErr)

print('Root Mean Square Error:', rootMeanSqErr)
print('Mean Absolute Error:', meanAbErr)
print('R-squared:', mlr.score(X_test, y_test))

# Scatter plots to visualize the relationships
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
axs = axs.flatten()

dependent_attrs = ['Week', 'Campus_Lecture_attendance', ]

for i, attr in enumerate(dependent_attrs):
    axs[i].scatter(dataset[attr], dataset['Module_X_Lecture_attendance'], color='blue')
    axs[i].set_xlabel(attr)
    axs[i].set_ylabel('Module_X_Lecture_attendance')

plt.tight_layout()
plt.show()