import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor 

try:
    df = pd.read_csv("startup_stage_dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: startup_stage_dataset.csv not found.")
    exit()

print(df.columns)

# 1.1 Select and Separate Features and Target
# Target (y): team_size (continuous numerical value)
y = df['team_size']

# Features (X): All other metrics + the categorical stage
feature_cols_to_drop = ['team_size', 'stage_num'] # Drop the target and the redundant numerical stage
X = df.drop(columns=feature_cols_to_drop)
y=np.log1p(y)
print(f"Initial features: {X.columns.tolist()}")
print(y.head)

X = pd.get_dummies(X, columns=['stage'], drop_first=True)
print(f"Features after One-Hot Encoding: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} test samples.")


# 1.3 Data Scaling (Normalization)
# Initialize StandardScaler and fit it only on the training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")


# 2.2 Select a Model - Using Random Forest Regressor
# Initialize the Random Forest Regressor
# Using 150 trees and limiting depth to reduce overfitting and compare to XGBoost settings.
rf_model = RandomForestRegressor(
    n_estimators=10,               # Number of trees in the forest
    max_depth=8,                   # Max depth of trees
    random_state=42,
    min_samples_split=10,
    max_features=1,
    n_jobs=-1
)

# --- 3. Training and Evaluation ---

# 3.1 Train the Model
print("\nStarting model training (RandomForestRegressor)...")
rf_model.fit(X_train_scaled, y_train)
print("Training complete.")

# 3.2 Evaluate Performance
# Make predictions on the scaled test data
y_pred_log = rf_model.predict(X_test_scaled)
y_pred_orig = np.expm1(y_pred_log)
y_test_orig_rf = np.expm1(y_test) 


# Calculate regression metrics
rf_rmse = np.sqrt(mean_squared_error(y_test_orig_rf, y_pred_orig))
rf_mae = mean_absolute_error(y_test_orig_rf, y_pred_orig)
rf_r2 = r2_score(y_test_orig_rf, y_pred_orig)

print(f"\n--- Model Evaluation Results (Target: team_size) ---")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f} team_size")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f} team_size")
print(f"R-squared (R^2) Score: {rf_r2:.4f} (Closer to 1.0 is better)")

# Ensure predictions are logical (no negative employees)
# In case the model predicts fractional team sizes, we usually round the prediction
y_pred_rounded = np.round(y_pred_orig)
mae_rounded = mean_absolute_error(y_test_orig_rf, y_pred_rounded)
print(f"\nMAE with rounded predictions: {mae_rounded:.2f} employees")

# --- 4. Model Interpretation and Application ---

# 4.1 Feature Importance (Unique to Tree-based models)
feature_importances = pd.Series(
    rf_model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

print("\n--- Feature Importances (Random Forest) ---")
print(feature_importances.head(5))

# 4.2 Deployment/Use Example: Predicting team size for a new startup
print("\n--- Example Prediction ---")
new_startup_metrics = {
    'months_since_foundation': 48,
    'funding_total_usd': 50000000.0,
    'funding_rounds': 6,
    'monthly_active_users': 1500000.0,
    'revenue_mrr_usd': 500000.0,
    'monthly_growth_rate': 0.10,
    'stage': 'growth'
}

# Convert new data to DataFrame
new_data_df = pd.DataFrame([new_startup_metrics])

# Replicate One-Hot Encoding structure
all_features = X.columns.tolist()
for col in all_features:
    if col not in new_data_df.columns:
        new_data_df[col] = 0

new_data_df = pd.get_dummies(new_data_df, columns=['stage'], drop_first=True)
new_data_df = new_data_df[X.columns] 

# Convert, scale, and transform
new_data_np = new_data_df.values.astype(np.float64)
new_data_scaled = scaler.transform(new_data_np)

# Predict the team size (output is log-transformed)
predicted_log = rf_model.predict(new_data_scaled)[0]
    
# CRITICAL: Inverse transform the prediction back to the original scale
predicted_team_size = np.expm1(predicted_log)

print(f"Input Metrics: {new_startup_metrics}")
print(f"\nPredicted Team Size (Estimated): **{predicted_team_size:.0f} employees**")
