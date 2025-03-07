import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Function to train and evaluate Random Forest
def train_and_evaluate_rf(n_estimators, X_train, X_test, y_train, y_test, fluids_test):
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, bootstrap=True)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Inverse transform log-transformed values
    actual_Rex = np.exp(X_test[:, 0])  
    predicted_Nux = np.exp(y_pred)     
    actual_Nux = np.exp(y_test)        
    
    # Calculate percentage errors for each sample
    error_percentages = np.abs((actual_Nux - predicted_Nux) / actual_Nux) * 100
    
    # Compute the 95th percentile of the error
    percentile_95_error = np.percentile(error_percentages, 95)
    
    return percentile_95_error

# Load dataset
df = pd.read_excel("https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx", engine='openpyxl')

# Convert categorical data to numerical
df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)

# Ensure the dataset has a 'Fluid' column
if 'Fluid' not in df.columns:
    raise ValueError("Dataset must contain a 'Fluid' column to categorize fluids.")

# Define features and target variable
X = df[['Rex', 'Pr', 'WallBC']].values
y = df['Nux'].values

# Log transformation
X[:, 0] = np.log10(X[:, 0])  # Log transform Rex
y = np.log10(y)              # Log transform Nux

# Train-test split
X_train, X_test, y_train, y_test, fluids_train, fluids_test = train_test_split(
    X, y, df['Fluid'].values, test_size=0.2, random_state=42
)

# Define the range of n_estimators
n_estimators_values = range(100, 1500, 100)

# Store 95th percentile errors for each n_estimators
percentile_95_errors = []

# Loop over different n_estimators values
for n in n_estimators_values:
    percentile_95_error = train_and_evaluate_rf(n, X_train, X_test, y_train, y_test, fluids_test)
    percentile_95_errors.append(percentile_95_error)

# Plotting the 95th percentile errors vs. n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, percentile_95_errors, marker='o', color='b', linestyle='-', linewidth=2)
plt.xlabel('n_estimators')
plt.ylabel('95th Percentile Error (%)')
plt.title('95th Percentile Error vs n_estimators for Random Forest')
plt.grid(True)
plt.show()
