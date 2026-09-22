import pandas as pd
import numpy as np
# from sklearn.model_selection import GridSearchCV # No longer needed as best params are set directly
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

# Define numerical and categorical features (consistent with prep.py and EDA)
numerical_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation', 'CityTier', 'Passport', 'OwnCar'
]

# --- Preprocessing Steps (retained for robustness beyond prep.py's cleaning) ---

# Outlier handling for numerical features (capping using insights from EDA)
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
    ],
    remainder='passthrough' # Keep any other columns not explicitly transformed
)

# --- Use the best performing XGBoost hyperparameters from previous execution ----
# Best parameters found: {'xgbclassifier__learning_rate': 0.2, 'xgbclassifier__max_depth': 7, 'xgbclassifier__n_estimators': 200}
best_xgb_params = {
    'learning_rate': 0.2,
    'max_depth': 7,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42
}

# Create the full pipeline with preprocessor and the best XGBoost classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(**best_xgb_params))
])

# Train the final model with the best hyperparameters (no GridSearchCV needed)
print("Training final XGBoost model with best hyperparameters...")
pipeline.fit(X_train, y_train)

# The trained pipeline is now our best model
best_model = pipeline

print(f"\nXGBoost Model Trained with Parameters: {best_xgb_params}")

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
    # Log the explicitly set best parameters
    mlflow.log_params(best_xgb_params)
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
