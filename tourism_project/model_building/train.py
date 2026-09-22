import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Load data
X_train = pd.read_csv('Xtrain.csv')
X_test = pd.read_csv('Xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze() # Use squeeze to convert to Series
y_test = pd.read_csv('ytest.csv').squeeze()   # Use squeeze to convert to Series

# Define numerical and categorical features
numerical_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation', 'CityTier', 'Passport', 'OwnCar'
]

# --- Preprocessing Steps ---

# 1. Handle Gender inconsistency (map unknown values to 'Male', the majority class)
def clean_gender(df):
    if 'Gender' in df.columns:
        # Standardize capitalization and remove whitespace
        df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
        # Map any value that is not 'Male' or 'Female' to 'Male' (majority class assumption)
        df.loc[~df['Gender'].isin(['Male', 'Female']), 'Gender'] = 'Male'
    return df

X_train = clean_gender(X_train)
X_test = clean_gender(X_test)

# 2. Outlier handling for numerical features (capping using insights from EDA)
def cap_outliers(df):
    # DurationOfPitch: Cap at 60 minutes
    df['DurationOfPitch'] = np.clip(df['DurationOfPitch'], None, 60)

    # NumberOfTrips: Cap at 15 trips
    df['NumberOfTrips'] = np.clip(df['NumberOfTrips'], None, 15)

    # MonthlyIncome: Cap using IQR (from previous EDA analysis)
    Q1 = 20751.0 # Approximate 25th percentile
    Q3 = 25301.0 # Approximate 75th percentile
    IQR = Q3 - Q1 # 4550
    lower_bound = Q1 - 1.5 * IQR # 13926.0
    upper_bound = Q3 + 1.5 * IQR # 32126.0
    df['MonthlyIncome'] = np.clip(df['MonthlyIncome'], lower_bound, upper_bound)
    return df

X_train = cap_outliers(X_train)
X_test = cap_outliers(X_test)

# Create preprocessing pipelines for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' handles unseen categories in test set gracefully
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline with preprocessor and XGBoost classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42))
])

# Define hyperparameter grid for GridSearchCV
# Parameters are slightly reduced for faster execution in a CI/CD pipeline context
param_grid = {
    'classifier__n_estimators': [50, 100],  # Number of boosting rounds
    'classifier__learning_rate': [0.05, 0.1], # Step size shrinkage
    'classifier__max_depth': [3, 5],        # Maximum depth of a tree
    'classifier__subsample': [0.7, 1.0],    # Subsample ratio of the training instance
    'classifier__colsample_bytree': [0.7, 1.0] # Subsample ratio of columns when constructing each tree
}

# Perform GridSearchCV for hyperparameter tuning
print("Starting GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1) # n_jobs=-1 uses all available cores
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best F1 score from cross-validation: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- MLflow Tracking ---
mlflow.set_tracking_uri("file:./mlruns") # Set local MLflow tracking URI
mlflow.set_experiment("Tourism Package Prediction") # Set the experiment name

with mlflow.start_run():
    print("\nLogging parameters and metrics to MLflow...")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    # Log the best model using mlflow.sklearn
    mlflow.sklearn.log_model(best_model, "best_xgboost_model")
    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
    print(f"Model logged to MLflow artifacts in run {mlflow.active_run().info.run_id}")

# --- Save the best model for deployment ---
deployment_path = 'tourism_project/deployment/best_model.joblib'
joblib.dump(best_model, deployment_path)
print(f"\nBest model saved locally for deployment to {deployment_path}")
