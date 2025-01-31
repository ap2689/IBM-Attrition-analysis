import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression

# Load data
st.title("🚀 Advanced IBM Employee Attrition Analysis")
st.write("Interactive HR dashboard with predictive analysis.")

# Read CSV
df = pd.read_csv("IBM.csv")

# Show dataset
st.subheader("📊 Sample Data")
st.dataframe(df.head())

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_department = st.sidebar.selectbox("Select Department", df["Department"].unique())
filtered_df = df[df["Department"] == selected_department]

# Attrition Distribution
st.subheader("📉 Attrition Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Attrition", data=df, ax=ax, palette="coolwarm")
st.pyplot(fig)

# Age vs Monthly Income
st.subheader("📈 Age vs Monthly Income")
fig, ax = plt.subplots()
sns.scatterplot(x="Age", y="MonthlyIncome", hue="Attrition", data=df, alpha=0.7)
st.pyplot(fig)

# Attrition by Department
st.subheader("🏢 Attrition by Department")
fig, ax = plt.subplots()
sns.countplot(x="Department", hue="Attrition", data=df, ax=ax, palette="viridis")
plt.xticks(rotation=45)
st.pyplot(fig)

# Machine Learning - Predict Attrition
st.subheader("🤖 Predict Employee Attrition")

# Prepare data for ML model
df_ml = df.copy()
df_ml = df_ml.dropna()  # Handle missing values
df_ml = pd.get_dummies(df_ml, drop_first=True)  # Convert categorical to numerical

X = df_ml.drop("Attrition_Yes", axis=1, errors='ignore')
y = df_ml["Attrition_Yes"] if "Attrition_Yes" in df_ml else df_ml.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show Accuracy
st.write(f"✅ Model Accuracy: **{accuracy:.2%}**")

# Classification Report
st.text("📌 Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("This model predicts whether an employee will leave the company based on factors like age, salary, work-life balance, and department.")

# Machine Learning - Predict Salary
st.subheader("💸 Predict Employee Salary")

# Prepare data for ML model
df_ml_salary = df.copy()
df_ml_salary = df_ml_salary.dropna(subset=["MonthlyIncome"])  # Drop rows where Salary is missing
df_ml_salary = pd.get_dummies(df_ml_salary, drop_first=True)  # Convert categorical to numerical
st.write(df_ml_salary.columns)  # Display all column names after dummy encoding

X_salary = df_ml_salary.drop("MonthlyIncome", axis=1)
y_salary = df_ml_salary["MonthlyIncome"]

X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

# Train model for salary prediction
salary_model = LinearRegression()
salary_model.fit(X_train_salary, y_train_salary)

# Predict salary
y_pred_salary = salary_model.predict(X_test_salary)
salary_accuracy = salary_model.score(X_test_salary, y_test_salary)

# Show Salary Prediction Accuracy
st.write(f"✅ Model Accuracy for Salary Prediction: **{salary_accuracy:.2%}**")

# Predict salary for a random employee
sample_employee = X_test_salary.sample(1)
predicted_salary = salary_model.predict(sample_employee)
st.write(f"Predicted Salary for a sample employee: **${predicted_salary[0]:,.2f}**")


