
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Loading and Inspection
# Load dataset
df = pd.read_csv("../Salary_Data.csv")
print(df.head())

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 2. Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# 3. Exploratory Data Analysis (EDA)
print("\nStatistical Summary:")
print(df.describe())
# 4. Data Visualization
# Histogram of Salary
plt.hist(df['Salary'])
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Distribution of Salary")
plt.show()

# Scatter plot to visualize relationship
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")
plt.show()
#Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 5. Predictive Modeling
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

# 6. Model Evaluation
print("\nModel Evaluation Metrics:")
print("Intercept:", lin_reg.intercept_)
print("Coefficient:", lin_reg.coef_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, label="Actual Salary")
plt.plot(X_test, y_pred, color='red', label="Predicted Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary Prediction")
plt.legend()
plt.show()



