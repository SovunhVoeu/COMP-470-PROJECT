!pip install pandas
!pip install numpy
!pip install seaborn
!pip install scikit-learn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("C:/Users/Brand/Downloads/weatherHistory.csv")

data.head()

data.tail()

data.describe()

# Define X and y correctly
X = data_sample.drop(['Loud Cover', 'Apparent Temperature (C)', 'Formatted Date', 'Summary', 'Precip Type', 'Daily Summary'], axis = 1)
y = data_sample["Loud Cover"]

# Convert specific columns to numeric, coercing errors to NaN
data_sample['Temperature (C)'] = pd.to_numeric(data_sample['Temperature (C)'], errors='coerce')
data_sample['Humidity'] = pd.to_numeric(data_sample['Humidity'], errors='coerce')
data_sample['Wind Speed (km/h)'] = pd.to_numeric(data_sample['Wind Speed (km/h)'], errors='coerce')
data_sample['Apparent Temperature (C)'] = pd.to_numeric(data_sample['Apparent Temperature (C)'], errors='coerce')
data_sample['Wind Bearing (degrees)'] = pd.to_numeric(data_sample['Wind Bearing (degrees)'], errors='coerce')
data_sample['Visibility (km)'] = pd.to_numeric(data_sample['Visibility (km)'], errors='coerce')
data_sample['Loud Cover'] = pd.to_numeric(data_sample['Loud Cover'], errors='coerce')
data_sample['Pressure (millibars)'] = pd.to_numeric(data_sample['Pressure (millibars)'], errors='coerce')

# Drop rows with NaN values
data_sample = data_sample.dropna()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

plt.figure(figsize=(20, 10))
plot_tree(rf_classifier.estimators_[0], filled=True, feature_names=X.columns, class_names=['Not Loud', 'Loud'], rounded=True, proportion=True)
plt.title("Visualization of First Decision Tree in Random Forest")
plt.show()

data_sample = data.drop(['Loud Cover', 'Apparent Temperature (C)', 'Formatted Date', 'Summary', 'Precip Type', 'Daily Summary'], axis = 1)

sns.pairplot(data_sample)

# Create a box plot
sns.boxplot(x='Wind Speed (km/h)', y='Temperature (C)', data=data)

# Display the plot
plt.show

import seaborn as sns
import matplotlib.pyplot as plt

# Box plot to detect outliers in 'Temperature (C)' column
sns.boxplot(x=data_sample['Temperature (C)'])
plt.show()

column = 'Temperature (C)'

# Calculate Variance
variance = data_sample[column].var()
print(f"Variance: {variance}")

# Calculate Standard Deviation
std_dev = data_sample[column].std()
print(f"Standard Deviation: {std_dev}")

# Calculate Interquartile Range (IQR)
Q1 = data_sample[column].quantile(0.25)  # 25th percentile
Q3 = data_sample[column].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1
print(f"Interquartile Range (IQR): {IQR}")

data_sample.head()

data_sample.describe()
