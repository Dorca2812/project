# 📦 Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# 📥 Load dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print("Initial Data Overview:")
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 📊 EDA: Department-wise attrition
plt.figure(figsize=(8, 4))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.tight_layout()
plt.show()

# 📊 EDA: Salary bands
df['SalaryBand'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(8, 4))
sns.countplot(x='SalaryBand', hue='Attrition', data=df)
plt.title('Attrition by Salary Band')
plt.tight_layout()
plt.show()

# 📊 EDA: Promotions
plt.figure(figsize=(8, 4))
sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=df)
plt.title('Years Since Last Promotion vs. Attrition')
plt.tight_layout()
plt.show()

# 🧹 Preprocessing
df_model = pd.get_dummies(df.drop(['EmployeeNumber'], axis=1), drop_first=True)
X = df_model.drop('Attrition_Yes', axis=1)
y = df_model['Attrition_Yes']

# 📤 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Use TreeExplainer
X_sample = X_train.sample(100, random_state=42)

# KernelExplainer works with any model
explainer = shap.KernelExplainer(model.predict, X_sample)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test, nsamples=100)

# Convert to array and check shape
shap_array = np.array(shap_values)
print("SHAP shape:", shap_array.shape)
print("X_test shape:", X_test.shape)

# Plot summary if shapes match
if shap_array.shape == X_test.shape:
    shap.summary_plot(shap_array, X_test)
else:
    print("⚠️ SHAP value shape mismatch. Plot skipped.")

# 📤 Export for Power BI
df['ModelPrediction'] = model.predict(pd.get_dummies(df.drop(['EmployeeNumber'], axis=1), drop_first=True).drop('Attrition_Yes', axis=1))
df.to_csv('processed_data_for_powerbi.csv', index=False)
# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🔍 Model Evaluation:")
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# 📝 Save Model Report
with open('model_report.txt', 'w') as f:
    f.write(f"Accuracy: {acc}\nConfusion Matrix:\n{cm}")
