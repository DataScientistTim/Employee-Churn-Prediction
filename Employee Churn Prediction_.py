#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[55]:


df = pd.read_csv(r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\Employee Churn Prediction\data\training_data.csv")
df.head()


# In[56]:


categorical_cols = ["Gender", "MaritalStatus", "AgeRange", "JobTitle", "HighestQualification", "EmploymentStatus"]


# In[57]:


df[categorical_cols] = df[categorical_cols].astype("category")


# In[58]:


# Convert EmploymentStatus into a binary target variable (0 = Active, 1 = Churned)
df["Churn"] = df["EmploymentStatus"].apply(lambda x: 1 if x in ["Churn"] else 0)

# Drop original EmploymentStatus column
df.drop(columns=["EmploymentStatus"], inplace=True)


# In[59]:


df.head()


# In[60]:


label_encoders = {}


# In[61]:


categorical_columns = ['Gender', 'MaritalStatus', 'AgeRange', 'JobTitle', 'HighestQualification']


# In[62]:


for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 


# In[63]:


joblib.dump(label_encoders, r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\Employee Churn Prediction\label_encoders.pkl")


# In[64]:


df.head()


# In[65]:


# Split dataset into training and testing sets
X = df.drop(columns=["EmployeeID", "Churn"])  # Drop EmployeeID as it's not relevant
y = df["Churn"]


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[67]:


# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[68]:


import joblib

joblib.dump(rf_model, r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\Employee Churn Prediction\my_model.pkl")


# In[69]:


data = pd.read_csv(r"C:\Users\wengt\Downloads\data.csv")  # Load new data
data.head()


# In[70]:


model = joblib.load(r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\Employee Churn Prediction\my_model.pkl")


# In[72]:


label_encoders = joblib.load(r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\Employee Churn Prediction\label_encoders.pkl")


# In[73]:


# Apply Label Encoding to categorical columns in new data
for col in categorical_columns:
    if col in data:
        data[col] = data[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)


# In[74]:


predictions = model.predict(data)


# In[75]:


data['Predictions'] = predictions


# In[ ]:


data.to_csv(r"C:\Users\wengt\Desktop\Glem\Hr Demo Set\code\prediction.csv", index=False)
print("Predictions saved to 'predictions.csv'")

