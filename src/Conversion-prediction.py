#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Install necessary libraries
import pandas as pd
import os


# In[5]:


os.chdir(r"C:\Users\HP\Desktop\Conversion-prediction")
print("Working directory:", os.getcwd())


# In[6]:


sessions = pd.read_csv("data/raw/website_sessions.csv")
orders = pd.read_csv("data/raw/orders_mav.csv")
pageviews = pd.read_csv("data/raw/website_pageviews.csv")


# In[7]:


# Print first few rows of dataset
print("Sessions: ")
print(sessions.head())


# In[8]:


print("Orders: ")
print(orders.head())


# In[9]:


print("Page views: ")
print(pageviews.head())


# In[16]:


# Understanding dataset shape
print("SESSIONS")
print(sessions.shape)

# Dataset information
sessions.info()

# Checking missing values per column
print("\nMissing values:")
print(sessions.isnull().sum())

# Total missing values
print("\nTotal missing values:")
print(sessions.isnull().sum().sum())


# In[17]:


print("Duplicate rows:", sessions.duplicated().sum())

print(
    "Duplicate session IDs:",
    sessions["website_session_id"].duplicated().sum()
)


# In[18]:


print("ORDERS")
print(orders.shape)
orders.info()

print("\nMissing values:")
print(orders.isnull().sum())

print("\nDuplicate rows:", orders.duplicated().sum())

print(
    "Duplicate session IDs:",
    orders["website_session_id"].duplicated().sum()
)


# In[19]:


print("PAGEVIEWS")
print(pageviews.shape)
pageviews.info()

print("\nMissing values:")
print(pageviews.isnull().sum())

print("\nDuplicate rows:", pageviews.duplicated().sum())

print(
    "Duplicate pageview IDs:",
    pageviews["website_pageview_id"].duplicated().sum()
)
print(
    "Unique sessions in pageviews:",
    pageviews["website_session_id"].nunique()
)


# In[20]:


# Check whether every order session exists in sessions
matching_sessions = orders["website_session_id"].isin(
    sessions["website_session_id"]
)

print("Matching order sessions:", matching_sessions.sum())
print("Unmatched order sessions:", (~matching_sessions).sum())


# In[21]:


# Merge website sessions with orders
df = sessions.merge(
    orders[["website_session_id"]],
    on="website_session_id",
    how="left",
    indicator=True
)

print("Merged dataset shape:", df.shape)
print(df.head())


# In[22]:


# Create the target variable
df["converted"] = (df["_merge"] == "both").astype(int)

# Remove the merge indicator
df.drop(columns=["_merge"], inplace=True)

# Check conversion distribution
print(df["converted"].value_counts())

print("\nConversion percentages:")
print(df["converted"].value_counts(normalize=True) * 100)


# In[23]:


# Handle missing marketing information
df["utm_source"] = df["utm_source"].fillna("Unknown")

df["utm_campaign"] = df["utm_campaign"].fillna("Unknown")

df["utm_content"] = df["utm_content"].fillna("Unknown")

df["http_referer"] = df["http_referer"].fillna("Direct/Unknown")


# In[24]:


print("Missing values after treatment:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print(df["converted"].isnull().sum())


# In[25]:


# Select modelling features
features = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer",
    "is_repeat_session"
]

# Keep the session timestamp for temporal splitting
model_df = df[
    ["website_session_id", "created_at"] + features + ["converted"]
].copy()

print(model_df.head())
print("\nModelling dataset shape:", model_df.shape)


# In[27]:


# Convert session timestamps
model_df["created_at"] = pd.to_datetime(
    model_df["created_at"],
    format="%m/%d/%Y %H:%M"
)

print(model_df["created_at"].dtype)


# In[28]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define categorical and numerical features
categorical_features = [
    "device_type",
    "utm_source",
    "utm_campaign",
    "utm_content",
    "http_referer"
]

numerical_features = [
    "is_repeat_session"
]

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

print("Preprocessing transformer created successfully.")


# In[29]:


# Sort records chronologically
model_df = model_df.sort_values("created_at").reset_index(drop=True)

# Define features
X = model_df[features]

# Define target
y = model_df["converted"]

# Retain timestamps for time-based splitting
timestamps = model_df["created_at"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# In[30]:


print(model_df.head())

print("\nDataset information:")
print(model_df.info())

print("\nMissing values:")
print(model_df.isnull().sum())


# In[31]:


# Create processed data directory
os.makedirs("data/processed", exist_ok=True)

# Save prepared dataset
model_df.to_csv(
    "data/processed/conversion_model_dataset.csv",
    index=False
)

print("Modelling dataset saved successfully.")


# In[32]:


# Calculate the index for an 80/20 time-based split
split_index = int(len(model_df) * 0.8)

# Split the dataset chronologically
train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

# Separate features (X) and target (y)
X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["converted"]
y_test = test_df["converted"]

# Display dataset sizes
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# In[33]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Build the machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Logistic Regression model created successfully.")


# In[34]:


# Train the model using the training data
model.fit(X_train, y_train)

print("Model training completed successfully.")


# In[35]:


# Generate predictions
y_pred = model.predict(X_test)

# Generate predicted conversion probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")

print("\nFirst 10 predicted outcomes:")
print(y_pred[:10])


# In[36]:


import pandas as pd

# Create a comparison DataFrame
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Conversion_Probability": y_prob
})

# Display the first 15 comparisons
print(results.head(15))


# In[37]:


# Identify correct and incorrect predictions
results["Correct_Prediction"] = (
    results["Actual"] == results["Predicted"]
)

print("Correct predictions:",
      results["Correct_Prediction"].sum())

print("Incorrect predictions:",
      (~results["Correct_Prediction"]).sum())

print("\nPrediction comparison:")
print(results["Correct_Prediction"].value_counts())


# In[38]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[39]:


# Check the distribution of actual conversions
print("Training conversion rate:")
print(y_train.mean())

print("\nTesting conversion rate:")
print(y_test.mean())

# Examine predicted purchase probabilities
print("\nPredicted probability statistics:")
print(pd.Series(y_prob).describe())

# Check the highest predicted probability
print("\nMaximum predicted purchase probability:")
print(y_prob.max())

# Check whether any probabilities reach 0.5
print("\nPredictions with probability >= 0.5:")
print((y_prob >= 0.5).sum())


# In[40]:


# Split the original training data chronologically
validation_index = int(len(train_df) * 0.8)

train_inner = train_df.iloc[:validation_index].copy()
val_df = train_df.iloc[validation_index:].copy()

X_fit = train_inner[features]
y_fit = train_inner["converted"]

X_val = val_df[features]
y_val = val_df["converted"]

print("Model training data:", X_fit.shape)
print("Validation data:", X_val.shape)

print("\nTraining conversion rate:", y_fit.mean())
print("Validation conversion rate:", y_val.mean())


# In[41]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

# Create an improved model
balanced_model = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train the model
balanced_model.fit(X_fit, y_fit)

print("Balanced Logistic Regression trained successfully.")


# In[42]:


# Generate validation predictions
val_pred = balanced_model.predict(X_val)

# Evaluate performance
print("Validation Accuracy:", accuracy_score(y_val, val_pred))

print("\nClassification Report:")
print(classification_report(y_val, val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred))


# In[43]:


# Get probabilities of purchase from the balanced model
val_prob = balanced_model.predict_proba(X_val)[:, 1]

print("Minimum probability:", val_prob.min())
print("Maximum probability:", val_prob.max())
print("Average probability:", val_prob.mean())


# In[44]:


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall at different thresholds
precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

# Calculate F1-score for each threshold
f1_scores = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-10)
)

# Find the threshold with the highest F1-score
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("Best threshold:", best_threshold)
print("Best validation F1-score:", f1_scores[best_index])
print("Precision at best threshold:", precision[best_index])
print("Recall at best threshold:", recall[best_index])


# In[45]:


# Apply the selected threshold
val_pred_tuned = (val_prob >= best_threshold).astype(int)

print("Tuned Validation Results")
print("\nClassification Report:")
print(classification_report(y_val, val_pred_tuned))

print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred_tuned))


# In[46]:


print("Best threshold:", best_threshold)
print("Best F1-score:", f1_scores[best_index])


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix

# Predict purchase probabilities on the test set
test_prob = balanced_model.predict_proba(X_test)[:, 1]

# Apply the selected validation threshold
test_pred = (test_prob >= best_threshold).astype(int)

# Evaluate final model
print("Final Test Classification Report:")
print(classification_report(y_test, test_pred))

print("Final Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))


# In[48]:


# Extract feature names and model coefficients
feature_names = balanced_model.named_steps["preprocessor"].get_feature_names_out()

coefficients = balanced_model.named_steps["classifier"].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print(feature_importance.head(15))


# In[49]:


import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": balanced_model,
    "threshold": best_threshold
}, "models/logistic_regression_model.pkl")

print("Model saved successfully.")


# In[ ]:




