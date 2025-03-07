import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_excel("https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx", engine='openpyxl')

df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)
X = df[['Rex', 'Pr', 'WallBC', 'c']].values
y = df['Nux'].values

#Log transformation
X[:, 0] = np.log10(X[:, 0])  # Rex
y = np.log10(y)              # Nux


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=1000, random_state=42, bootstrap=True)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

#Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Inverse transform the scaled Rex values for plotting
actual_Rex = np.exp(X_test[:, 0])  # Inverse transform the log-transformed Rex
predicted_Nux = np.exp(y_pred)      # Inverse transform the predicted log-transformed Nux
actual_Nux = np.exp(y_test)          # Inverse transform the actual log-transformed Nux

#Plotting
plt.figure(figsize=(10, 6))
plt.scatter(actual_Rex, actual_Nux, color='blue', label='Actual Nux', alpha=0.5, marker='o')
plt.scatter(actual_Rex, predicted_Nux, color='orange', label='Predicted Nux', alpha=0.5, marker='x')
plt.xlabel('Rex')
plt.ylabel('Nux')
plt.title('Random Forest Predictions vs Actual Nux Values')
plt.legend()
plt.grid()
plt.show()