import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data import & preparation
df = pd.read_excel(
    "https://raw.githubusercontent.com/adhicurry/ML_HeatTransfer_Fluids/main/FlatPlate_Models/Data_Combined_1.xlsx",
    engine='openpyxl'
)
df['WallBC'] = df['WallBC'].replace({'UWT': 0, 'UHF': 1}).astype(float)
data_subset = df[df['Figure'].isin([2, 3, '4a', '4b', '4c', '6a', '6b', 8, '9a'])].reset_index(drop=True)

# Define features and target; apply log10 transformation for scaling
X = data_subset[['Rex', 'Pr', 'WallBC', 'c']].values
y = data_subset['Nux'].values
X[:, 0] = np.log10(X[:, 0])
y = np.log10(y)

# Stratified splitting using the 'Figure' column
figures = data_subset['Figure'].astype(str).values
X_train_temp, X_test, y_train_temp, y_test, fig_train_temp, fig_test = train_test_split(
    X, y, figures, test_size=0.20, stratify=figures, shuffle=True, random_state=42
)
X_train, X_val, y_train, y_val, fig_train, fig_val = train_test_split(
    X_train_temp, y_train_temp, fig_train_temp, test_size=0.15, stratify=fig_train_temp, shuffle=True, random_state=42
)

# Scale features and target
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
joblib.dump(X_scaler, 'X_scaler.save')
joblib.dump(y_scaler, 'y_scaler.save')

# Convert arrays to PyTorch tensors and define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# Custom dataset that returns indices
class CustomTrainDataset(Dataset):
    def __init__(self, X, y):
        """
        X and y should be numpy arrays or tensors on CPU.
        """
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.cpu()
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y.cpu()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], idx

train_dataset = CustomTrainDataset(X_train_scaled, y_train_scaled)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Heteroscedastic expert model
class HeteroscedasticResidualMLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_blocks=4, dropout_rate=0.10):
        """
        Residual MLP with global skip connections that outputs the predicted mean and log variance.
        """
        super(HeteroscedasticResidualMLP, self).__init__()
        self.linear_skip = nn.Linear(input_size, 1)
        self.linear_skip_var = nn.Linear(input_size, 1)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_layer_var = nn.Linear(hidden_size, 1)

    def forward(self, x):
        skip_mean = self.linear_skip(x)
        skip_var = self.linear_skip_var(x)
        out = self.input_layer(x)
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual
        mean = self.output_layer(out) + skip_mean
        log_var = torch.clamp(self.output_layer_var(out) + skip_var, min=-10, max=10)
        return mean, log_var

# Mixture-of-experts model
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_blocks=4, dropout_rate=0.10):
        """
        Mixture-of-experts model with two experts:
          - general_expert: trained on the full data
          - transition_expert: specialized for the transition region
        A gating network learns to weight the experts' outputs.
        """
        super(MixtureOfExperts, self).__init__()
        self.general_expert = HeteroscedasticResidualMLP(input_size, hidden_size, num_blocks, dropout_rate)
        self.transition_expert = HeteroscedasticResidualMLP(input_size, hidden_size, num_blocks, dropout_rate)
        self.gating = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.gating(x)
        mean_general, log_var_general = self.general_expert(x)
        mean_transition, log_var_transition = self.transition_expert(x)
        mean = weight * mean_transition + (1 - weight) * mean_general
        log_var = weight * log_var_transition + (1 - weight) * log_var_general
        return mean, log_var

# Heteroscedastic loss function
def heteroscedastic_loss(pred_mean, pred_log_var, target):
    """
    Negative log likelihood loss for a Gaussian:
      loss = 0.5 * log_var + 0.5 * ((target - pred_mean)^2 / exp(log_var))
    """
    precision = torch.exp(-pred_log_var)
    loss = 0.5 * pred_log_var + 0.5 * ((target - pred_mean) ** 2 * precision)
    return loss

# Training setup
model = MixtureOfExperts(input_size=4, hidden_size=128, num_blocks=4, dropout_rate=0.10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=150, verbose=True)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
num_epochs = 5000
early_stop_patience = 500
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
train_losses = []
val_losses = []

# Extract scaling parameters for log10(Rex)
scale0 = float(X_scaler.scale_[0])
mean0 = float(X_scaler.mean_[0])

print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y, _ in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred_mean, pred_log_var = model(batch_x)
        loss_per_sample = heteroscedastic_loss(pred_mean, pred_log_var, batch_y)

        # Recover original Rex for weighting
        log10_Rex = batch_x[:, 0] * scale0 + mean0
        Rex = 10 ** log10_Rex
        transition_mask = (Rex >= 1e5) & (Rex <= 5e5)
        sample_weights = torch.where(transition_mask, torch.tensor(2.0, device=device),
                                     torch.tensor(1.0, device=device))
        weighted_loss = (sample_weights * loss_per_sample.squeeze()).mean()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += weighted_loss.item() * batch_x.size(0)

    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            pred_mean_val, pred_log_var_val = model(val_x)
            val_loss = heteroscedastic_loss(pred_mean_val, pred_log_var_val, val_y).mean().item()
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

print(f"\nBest validation loss: {best_val_loss:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Mixture-of-Experts Model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

torch.save(best_model_state, 'best_MixtureOfExperts_Model.pth')
model.load_state_dict(best_model_state)

# Evaluation & plotting
def calculate_metrics(predictions, actuals):
    error_pct = 100 * np.abs(predictions - actuals) / actuals
    return {
        'MAE': np.mean(np.abs(predictions - actuals)),
        'Average_Percentage_Error': np.mean(error_pct),
        'Min_Error': np.min(error_pct),
        'Max_Error': np.max(error_pct),
        'Median_Error': np.median(error_pct),
        'Percentile_68': np.percentile(error_pct, 68),
        'Percentile_95': np.percentile(error_pct, 95),
        'Percentile_99': np.percentile(error_pct, 99)
    }

model.eval()
with torch.no_grad():
    preds_mean, preds_log_var = model(X_test_tensor)
    preds = preds_mean.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

# Inverse-transform predictions and targets back to original values
preds_log = y_scaler.inverse_transform(preds)
y_test_log = y_scaler.inverse_transform(y_test_np)
preds_original = 10 ** preds_log
y_test_original = 10 ** y_test_log

metrics = calculate_metrics(preds_original, y_test_original)
print("\nMixture-of-Experts Model Evaluation Metrics (Overall):")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

substance_mapping = {
    '2': 'Air',
    '8': 'Air',
    '3': 'Water',
    '6a': 'Water',
    '6b': 'Water',
    '4a': 'Oil',
    '4b': 'Oil',
    '4c': 'Oil',
    '9a': 'Oil'
}
substances = np.array([substance_mapping[str(fig)] for fig in fig_test])
unique_substances = np.unique(substances)

print("\nError Metrics per Substance:")
for substance in unique_substances:
    mask = (substances == substance)
    preds_sub = preds_original[mask]
    actuals_sub = y_test_original[mask]
    metrics_sub = calculate_metrics(preds_sub, actuals_sub)
    print(f"\nSubstance: {substance}")
    for k, v in metrics_sub.items():
        print(f"{k}: {v:.2f}")

X_test_unscaled = X_scaler.inverse_transform(X_test_scaled)
Rex_test = 10 ** X_test_unscaled[:, 0]
plt.figure(figsize=(10, 6))
plt.loglog(Rex_test, y_test_original, 'o', label='Actual Data')
plt.loglog(Rex_test, preds_original, 'x', label='Predictions (Mixture-of-Experts)')
plt.xlabel('Rex')
plt.ylabel('Nux')
plt.title('Predictions vs Actual (Mixture-of-Experts Model)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()
