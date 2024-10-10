import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the data
data_dir = Path('./data')
train_df = pd.read_csv(data_dir/'train.csv')
test_df = pd.read_csv(data_dir/'test.csv')

# Drop rows where target 'sii' is missing
train_df = train_df.dropna(subset=['sii'])

# Separate features and target
X = train_df.drop(columns=['sii', 'id'])
y = train_df['sii']

# Get the common columns between train and test data
common_columns = list(set(X.columns) & set(test_df.columns) - {'id'})

# Update X to only include common columns
X = X[common_columns]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Impute numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical features
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss')

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0]
}

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on validation set
y_val_pred = best_model.predict(X_val)

# Evaluate accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Process test data
test_X = test_df[common_columns]

# Apply the same preprocessing to test data
test_X[num_cols] = num_imputer.transform(test_X[num_cols])
test_X[cat_cols] = cat_imputer.transform(test_X[cat_cols])

for col in cat_cols:
    test_X[col] = le.transform(test_X[col])

# Make predictions on test set
test_pred = best_model.predict(test_X)

# Prepare submission
submission = pd.DataFrame({'id': test_df['id'], 'sii': test_pred})
submission.to_csv('sample_submission.csv', index=False)

# Print some diagnostic information
print(f"Number of features used in training: {len(common_columns)}")
print(f"Features used: {common_columns}")
