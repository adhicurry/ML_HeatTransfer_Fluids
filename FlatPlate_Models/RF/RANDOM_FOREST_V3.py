import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel(
    "https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx",
    engine='openpyxl'
)

df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)

if 'Fluid' not in df.columns:
    raise ValueError("Dataset must contain a 'Fluid' column to categorize fluids.")

feature_columns = ['Rex', 'Pr', 'WallBC']
  

X = df[feature_columns].values
y = df['Nux'].values

X[:, 0] = np.log10(X[:, 0])  
y = np.log10(y)              

# Train-test split
X_train, X_test, y_train, y_test, fluids_train, fluids_test = train_test_split(
    X, y, df['Fluid'].values, test_size=0.2, random_state=42
)

# Train Random Forest model with best parameters
rf_model = RandomForestRegressor(
    max_depth=30,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=500,
    random_state=42,
    bootstrap=True
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Compute performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Inverse transform log-transformed values
actual_Rex = np.exp(X_test[:, 0])  
predicted_Nux = np.exp(y_pred)     
actual_Nux = np.exp(y_test)        

# Compute percentage errors for each sample
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

# Convert overall stats to a DataFrame and combine with fluid-specific stats
overall_error_df = pd.DataFrame(overall_error_stats, index=['Overall'])
combined_error_stats = pd.concat([fluid_error_stats, overall_error_df])

# Print the combined error statistics table
print("\nError Statistics by Fluid and Overall:")
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
