

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  accuracy_score,confusion_matrix,classification_report)

# 1. Data Loading
# Load Iris dataset
df = sns.load_dataset("iris")

# Inspect dataset
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
# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include='all'))

# 4. Data Visualization
# Line Plot
plt.plot(df.index, df['sepal_length'])
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.title('Line Plot of Sepal Length')
plt.show()

plt.plot(df.index, df['sepal_width'])
plt.xlabel('Index')
plt.ylabel('Sepal Width')
plt.title('Line Plot of Sepal Width')
plt.show()

plt.plot(df.index, df['petal_length'])
plt.xlabel('Index')
plt.ylabel('Petal Length')
plt.title('Line Plot of Petal Length')
plt.show()

plt.plot(df.index, df['petal_width'])
plt.xlabel('Index')
plt.ylabel('Petal Width')
plt.title('Line Plot of Petal Width')
plt.show()


# Histogram
sns.histplot(df["sepal_length"], kde=True, bins=20, color="blue")
plt.title("Distribution of Sepal Length")
plt.show()

sns.histplot(df["sepal_width"], kde=True, bins=20, color="red")
plt.title("Distribution of Sepal width")
plt.show()

sns.histplot(df["petal_length"], kde=True, bins=20, color="violet")
plt.title("Distribution of petal Length")
plt.show()

sns.histplot(df["petal_width"], kde=True, bins=20, color="brown")
plt.title("Distribution of petal width")
plt.show()
# Box Plot
sns.boxplot(x='species', y='sepal_length',color='blue', data=df)
plt.title('Box Plot of Sepal lenth by Species')
plt.show()

sns.boxplot(x='species', y='sepal_width',color='red', data=df)
plt.title('Box Plot of Sepal Width by Species')
plt.show()

sns.boxplot(x='species', y='petal_length',color='violet', data=df)
plt.title('Box Plot of petal length by Species')
plt.show()

sns.boxplot(x='species', y='petal_width',color='brown', data=df)
plt.title('Box Plot of petal width by Species')
plt.show()




# Scatter Plot
sns.scatterplot( x='sepal_length',y='petal_length',  hue='species',data=df)
plt.title('Scatter Plot of Sepal Length vs Petal Length')
plt.show()

sns.scatterplot( x='sepal_width',y='petal_width',  hue='species',data=df)
plt.title('Scatter Plot of Sepal width vs Petal width by Species')
plt.show()

# Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pair Plot
sns.pairplot(df, hue='species')
plt.suptitle("Pair Plot of Iris Features", y=1.02)
plt.show()

# 5. Predictive Modeling
# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
# Create Logistic Regression model
log_reg= LogisticRegression(max_iter=200)
# Train the model
log_reg.fit(X_train,y_train)
# Make predictions
y_pred = log_reg.predict(X_test)

# 6. Model Evaluation
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))