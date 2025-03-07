import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, max_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx", engine='openpyxl')

# Encode categorical variable
df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)

# Define features and target
X = df[['Rex', 'Pr', 'WallBC', 'c']].values  # Exclude index_col
y = df['Nux'].values

# Log transformation
X[:, 0] = np.log10(X[:, 0])  # Rex
y = np.log10(y)  # Nux

# Train-test split with indices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, df.index.to_numpy(), test_size=0.2, random_state=42
)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap=True)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Compute errors
test_actual = np.exp(y_test)
test_predicted = np.exp(y_pred)
errors = np.abs(test_actual - test_predicted)
percentage_errors = errors / test_actual * 100  

# Function to compute requested metrics
def compute_metrics(errors, percentage_errors):
    if len(errors) == 0:
        return {"Average_Percentage_Error": None, "MAE": None, "Max_Error": None,
                "Median_Error": None, "Min_Error": None, "Percentile_68": None,
                "Percentile_95": None, "Percentile_99": None}
    return {
        "Average_Percentage_Error": np.mean(percentage_errors),
        "MAE": np.mean(percentage_errors),  # Expressed as a percentage
        "Max_Error": np.max(percentage_errors),
        "Median_Error": np.median(percentage_errors),
        "Min_Error": np.min(percentage_errors),
        "Percentile_68": np.percentile(percentage_errors, 68),
        "Percentile_95": np.percentile(percentage_errors, 95),
        "Percentile_99": np.percentile(percentage_errors, 99)
    }

# Compute overall metrics
metrics_overall = compute_metrics(errors, percentage_errors)

# Compute metrics for each fluid type
test_c_values = df.loc[test_indices, 'c'].to_numpy()
metrics_air = compute_metrics(errors[test_c_values == 1], percentage_errors[test_c_values == 1]) if np.any(test_c_values == 1) else None
metrics_oil = compute_metrics(errors[test_c_values == 2], percentage_errors[test_c_values == 2]) if np.any(test_c_values == 2) else None
metrics_water = compute_metrics(errors[test_c_values == 3], percentage_errors[test_c_values == 3]) if np.any(test_c_values == 3) else None


metrics_df = pd.DataFrame({
    "Overall": metrics_overall,
    "air": metrics_air,
    "oil": metrics_oil,
    "water": metrics_water
})

print(metrics_df)

# Plot results
actual_Rex = np.exp(X_test[:, 0])  # Inverse transform Rex
plt.figure(figsize=(10, 6))
plt.scatter(actual_Rex, test_actual, color='blue', label='Actual Nux', alpha=0.5, marker='o')
plt.scatter(actual_Rex, test_predicted, color='orange', label='Predicted Nux', alpha=0.5, marker='x')
plt.xlabel('Rex')
plt.ylabel('Nux')
plt.title('Random Forest Predictions vs Actual Nux Values')
plt.legend()
plt.grid()
plt.show()
