#!/usr/bin/env python
# coding: utf-8

# # Health Score Model
# This notebook will guide you through the process of creating a health score model based on various health parameters. We will generate and simulate datasets for this purpose.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Define the number of samples and features
n_samples = 1000
n_features = 27

# Generate the dataset
X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)

# Define the feature names
feature_names = ['age', 'gender', 'smoking_history', 'alcohol_consumption', 'family_history_diabetes', 'family_history_hypertension',
                  'family_history_hyperlipidemia', 'BMI', 'blood_pressure', 'heart_rate_variability', 'blood_glucose_level',
                  'blood_LDL_level', 'blood_HDL_level', 'blood_triglyceride_level', 'blood_urea_nitrogen_level', 'creatinine_level',
                  'uric_acid_level', 'hemoglobin_level', 'hematocrit_level', 'MCV', 'platelet_count', 'AST_level', 'ALT_level',
                  'ALP_level', 'direct_bilirubin_level', 'cortisol_level', 'cardiomegaly_on_chest_radiography']

# Create the DataFrame
df = pd.DataFrame(X, columns=feature_names)

# Add the target variable
df['EKG_abnormality'] = y

# Display the first few rows of the DataFrame
df.head()

# ## Data Preprocessing
# Before we can use this data to train a model, we need to preprocess it. This includes scaling the features and splitting the data into training and test sets.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the features and the target
X = df.drop('EKG_abnormality', axis=1)
y = df['EKG_abnormality']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ## Model Selection and Training
# Now that we have preprocessed our data, we can train a model on it. We will use a Random Forest classifier for this task. Random Forest is a versatile and widely-used machine learning algorithm that can capture complex patterns in the data.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Print the classification report
print(classification_report(y_test, y_pred))

# ## Model Fine-Tuning
# To improve the performance of our model, we can fine-tune its hyperparameters. Hyperparameters are the parameters of the model that are not learned from the data, but are set by the user before training. For the Random Forest model, these include the number of trees in the forest (n_estimators) and the maximum depth of the trees (max_depth), among others.
# 
# We will use GridSearchCV to find the best hyperparameters for our model. GridSearchCV performs an exhaustive search over a specified parameter grid, and finds the parameters that give the best performance according to a scoring metric (in this case, accuracy).

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize the grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Get the best score
best_score = grid_search.best_score_
print(f'Best score: {best_score*100:.2f}%')

# ## Training the Final Model
# Now that we have found the best hyperparameters for our model, we can train a new Random Forest model using these parameters and evaluate it on the test set.

# In[ ]:


# Initialize the model with the best parameters
model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                               max_depth=best_params['max_depth'],
                               min_samples_split=best_params['min_samples_split'],
                               random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Print the classification report
print(classification_report(y_test, y_pred))

# ## Model Validation
# To further ensure the robustness of our model, we can perform cross-validation. Cross-validation is a technique where the training set is split into k smaller sets or 'folds'. The model is then trained on k-1 of these folds, and validated on the remaining fold. This process is repeated k times, with each fold used exactly once as the validation data. The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. This method gives a more robust estimate of the model's performance.

# In[ ]:


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation scores
print(f'CV scores: {cv_scores}')

# Print the mean cross-validation score
print(f'Mean CV score: {cv_scores.mean()*100:.2f}%')

# ## Making Predictions
# Now that we have a trained and validated model, we can use it to make predictions on new data. For example, we could input the health data of a new patient, and the model would predict whether they are likely to have an EKG abnormality.
# 
# Let's simulate a new patient's data and use our model to make a prediction.

# In[ ]:


import numpy as np

# Simulate a new patient's data
new_patient = np.array([[50,  # age
                         1,   # gender (1 = male, 0 = female)
                         0,   # smoking history (1 = yes, 0 = no)
                         0,   # alcohol consumption history (1 = yes, 0 = no)
                         0,   # family history of Diabetes Mellitus (1 = yes, 0 = no)
                         0,   # family history of Hypertension (1 = yes, 0 = no)
                         0,   # family history of Hyperlipidemia (1 = yes, 0 = no)
                         25,  # BMI
                         120, # blood pressure
                         60,  # heart rate variability
                         100, # blood glucose level
                         100, # blood LDL level
                         50,  # blood HDL level
                         150, # blood triglyceride level
                         15,  # blood urea nitrogen level
                         1,   # creatinine level
                         6,   # uric acid level
                         15,  # hemoglobin level
                         45,  # hematocrit level
                         90,  # MCV in complete blood count
                         250, # platelet count
                         20,  # AST level
                         20,  # ALT level
                         70,  # ALP level
                         0.2, # direct bilirubin level
                         10,  # cortisol level
                         0,   # cardiomegaly on chest radiography (1 = yes, 0 = no)
                         ]])

# Scale the new patient's data
new_patient = scaler.transform(new_patient)

# Use the model to make a prediction
prediction = model.predict(new_patient)

# Print the prediction
if prediction[0] == 1:
    print('The patient is likely to have an EKG abnormality.')
else:
    print('The patient is unlikely to have an EKG abnormality.')
