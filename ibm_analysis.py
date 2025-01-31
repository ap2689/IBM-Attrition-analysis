import pandas as pd

# Load the IBM.csv file
df = pd.read_csv('IBM.csv')  # Make sure IBM.csv is in the same folder as this script

# Print the first few rows of the dataset to see what it looks like
print(df.head())

# Check if there are any missing values in the dataset
print("\nMissing Values:")
print(df.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# 1. Distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, color="blue")
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Attrition by Department
plt.figure(figsize=(8, 6))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# 3. Monthly Income vs. Age (for attrition prediction)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='MonthlyIncome', hue='Attrition', data=df)
plt.title('Monthly Income vs Age (Attrition)')
plt.xlabel('Age')
plt.ylabel('Monthly Income')
plt.show()

# 4. WorkLifeBalance vs. Attrition
plt.figure(figsize=(8, 6))
sns.countplot(x='WorkLifeBalance', hue='Attrition', data=df)
plt.title('WorkLifeBalance vs. Attrition')
plt.xlabel('WorkLifeBalance')
plt.ylabel('Count')
plt.show()

# 5. YearsAtCompany vs. Attrition
plt.figure(figsize=(8, 6))
sns.countplot(x='YearsAtCompany', hue='Attrition', data=df)
plt.title('YearsAtCompany vs. Attrition')
plt.xlabel('YearsAtCompany')
plt.ylabel('Count')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

# Encode categorical columns (e.g., Attrition, Department, etc.) as numbers
label_encoder = LabelEncoder()

# Encode 'Attrition' column (Yes = 1, No = 0)
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# Encode other categorical columns (if necessary, like Department)
df['Department'] = label_encoder.fit_transform(df['Department'])

# Select relevant features for the prediction
X = df[['Age', 'DistanceFromHome', 'Education', 'MonthlyIncome', 'NumCompaniesWorked', 'WorkLifeBalance', 'YearsAtCompany', 'Department']]
y = df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred))
