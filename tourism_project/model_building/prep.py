import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('tourism_project/data/tourism.csv')

# Data Anomoly correction
# 1. Fix the Gender typos by replacing 'Fe Male' with 'Female'
data['Gender'] = data['Gender'].replace('Fe Male', 'Female')

# 2. Consolidate 'Unmarried' into 'Single' for cleaner demographic tracking
data['MaritalStatus'] = data['MaritalStatus'].replace('Unmarried', 'Single')

# Remove unnecessary columns (e.g., 'Unnamed: 0')
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])
data.drop(columns=['CustomerID'], inplace=True)

# Define features (X) and target (y)
X = data.drop('ProdTaken', axis=1)
y = data['ProdTaken']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the training and testing sets locally as CSV files
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

print('Data preparation complete. Training and testing sets saved as CSV files.')
