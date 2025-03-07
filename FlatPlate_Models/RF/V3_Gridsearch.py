import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Set up the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Define an expanded parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 400],
    'max_depth': [5, 10, 20, 30, 40, 50, None],  # Test a wider range of depths
    'min_samples_split': [2, 5, 10, 15],         # Wider range for splitting nodes
    'min_samples_leaf': [1, 2, 4, 6],            # Larger range for minimum leaf size
    'max_features': ['auto', 'sqrt', 'log2', 0.1, 0.2, 0.3, 0.4],  # Try different feature selections
    'bootstrap': [True, False]                   # Bootstrap sampling options
}

# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit GridSearchCV to find the best parameters
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Use the best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_rf_model.predict(X_test)

# Compute performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Inverse transform log-transformed values
actual_Rex = np.exp(X_test[:, 0])  
predicted_Nux = np.exp(y_pred)     
actual_Nux = np.exp(y_test)        

# Calculate percentage errors for each sample
error_percentages = np.abs((actual_Nux - predicted_Nux) / actual_Nux) * 100

# Store results in a DataFrame
results_df = pd.DataFrame({
    'Fluid': fluids_test,
    'Rex': actual_Rex,
    'Actual Nux': actual_Nux,
    'Predicted Nux': predicted_Nux,
    'Error (%)': error_percentages
})

# Compute error statistics for each fluid
fluid_error_stats = results_df.groupby('Fluid')['Error (%)'].agg(
    ['mean', 'median', 'min', 'max', 
     lambda x: np.percentile(x, 68),  # 68th percentile
     lambda x: np.percentile(x, 95),  # 95th percentile
     lambda x: np.percentile(x, 99)]  # 99th percentile
)
fluid_error_stats.columns = ['Mean', 'Median', 'Min', 'Max', '68th %', '95th %', '99th %']

# Compute overall error statistics
overall_error_stats = {
    'Mean': error_percentages.mean(),
    'Median': np.median(error_percentages),
    'Min': error_percentages.min(),
    'Max': error_percentages.max(),
    '68th %': np.percentile(error_percentages, 68),
    '95th %': np.percentile(error_percentages, 95),
    '99th %': np.percentile(error_percentages, 99)
}

# Add overall error statistics to the fluid error stats table
overall_error_df = pd.DataFrame(overall_error_stats, index=['overall'])
combined_error_stats = pd.concat([fluid_error_stats, overall_error_df])

# Print the combined error statistics table
print("\nError statistics by fluid and overall:")
print(combined_error_stats)

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(actual_Rex, actual_Nux, color='blue', label='Actual Nux', alpha=0.5, marker='o')
plt.scatter(actual_Rex, predicted_Nux, color='orange', label='Predicted Nux', alpha=0.5, marker='x')
plt.xlabel('Rex')
plt.ylabel('Nux')
plt.title('Random Forest Predictions vs Actual Nux Values')
plt.legend()
plt.grid()
plt.show()
