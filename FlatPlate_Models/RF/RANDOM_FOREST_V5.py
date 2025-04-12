import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load dataset
df = pd.read_excel(
    "https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx",
    engine='openpyxl'
)

df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)

# Ensure 'Fluid' column exists
if 'Fluid' not in df.columns:
    raise ValueError("Dataset must contain a 'Fluid' column to categorize fluids.")

# --- FEATURE ENGINEERING ---
# Calculate Peclet number and log1p transformations
df['Peclet'] = df['Rex'] * df['Pr']
df['logRex'] = np.log1p(df['Rex'])
df['logPe'] = np.log1p(df['Peclet'])

# Feature selection (including the new feature)
feature_columns = ['logRex', 'Pr', 'WallBC', 'logPe']
X = df[feature_columns].values
y = df['Nux'].values

# Apply log1p transformation to target
y = np.log1p(y)

# Use RobustScaler to handle outliers
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Stratified Split by Fluid Type
X_train, X_test, y_train, y_test, fluids_train, fluids_test = train_test_split(
    X, y, df['Fluid'].values, test_size=0.2, random_state=42, stratify=df['Fluid'].values
)

# Define the model
rf_model = RandomForestRegressor(
    n_estimators=1000, max_depth=20, max_features='log2',
    min_samples_leaf=2, min_samples_split=2, random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Cross-validation score
cv_score = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2').mean()
print(f"Cross-Validation R² Score: {cv_score:.4f}")

# Predictions
y_pred = rf_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Inverse transform log1p-transformed values
actual_logRex = X_test[:, 0]
actual_Rex = np.expm1(actual_logRex)
predicted_Nux = np.expm1(y_pred)
actual_Nux = np.expm1(y_test)

# Compute percentage errors
error_percentages = np.abs((actual_Nux - predicted_Nux) / actual_Nux) * 100

# Store results
results_df = pd.DataFrame({
    'Fluid': fluids_test,
    'Rex': actual_Rex,
    'Actual Nux': actual_Nux,
    'Predicted Nux': predicted_Nux,
    'Error (%)': error_percentages
})

# Error stats by fluid
fluid_error_stats = results_df.groupby('Fluid')['Error (%)'].describe(percentiles=[0.68, 0.95, 0.99])

# Overall error stats
overall_error_df = pd.DataFrame(pd.Series(error_percentages).describe(percentiles=[0.68, 0.95, 0.99])).T
overall_error_df.index = ['Overall']

# Combine
combined_error_stats = pd.concat([fluid_error_stats, overall_error_df])
print("\nError Statistics by Fluid and Overall:")
print(combined_error_stats)

# Scatter plot of predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=actual_Rex, y=actual_Nux, color='blue', label='Actual Nux', alpha=0.6)
sns.scatterplot(x=actual_Rex, y=predicted_Nux, color='orange', label='Predicted Nux', alpha=0.6)
plt.xlabel('Rex')
plt.ylabel('Nux')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.grid()
plt.show()

# Residual histogram
plt.figure(figsize=(10, 5))
sns.histplot(error_percentages, bins=50, kde=True, color='red')
plt.xlabel('Error (%)')
plt.title('Error Distribution')
plt.show()

# Error percentage vs Rex plot
plt.figure(figsize=(10, 6))
plt.semilogx(actual_Rex, error_percentages, '.', alpha=0.6)
plt.title('Prediction Error Percentage vs Rex')
plt.xlabel('Rex')
plt.ylabel('Error %')
plt.grid(True, alpha=0.2)
plt.show()

# Print detailed error statistics
mae = np.mean(np.abs(predicted_Nux - actual_Nux))
mean_pct_error = np.mean(error_percentages)
min_error = np.min(error_percentages)
max_error = np.max(error_percentages)
median_error = np.median(error_percentages)
error_68 = np.percentile(error_percentages, 68)
error_95 = np.percentile(error_percentages, 95)
error_99 = np.percentile(error_percentages, 99)

print(f"Test MAE: {mae:.6f}")
print(f"Average Percentage Error: {mean_pct_error:.2f}%")
print(f"Minimum Error: {min_error:.2f}%")
print(f"Maximum Error: {max_error:.2f}%")
print(f"Median Error: {median_error:.2f}%")
print(f"68% of the data lies within an error of {error_68:.2f}%")
print(f"95% of the data lies within an error of {error_95:.2f}%")
print(f"99% of the data lies within an error of {error_99:.2f}%")
