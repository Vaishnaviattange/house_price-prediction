# ===============================
# House Price Prediction Project
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("\n===============================")
print(" House Price Prediction Started ")
print("===============================\n")

# Load datasets
train = pd.read_csv("Cleaned train.csv")
test = pd.read_csv("Cleaned test.csv")

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

# Automatically detect target column
target_column = train.columns[-1]
print("Target column detected:", target_column)

# Separate features and target
X = train.drop(target_column, axis=1)
y = train[target_column]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluation (VERSION-SAFE)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Predict on test data
test_predictions = model.predict(test)

# Save predictions
output = pd.DataFrame({
    "Predicted_Price": test_predictions
})

output.to_csv("house_price_predictions.csv", index=False)

print("\nPrediction file saved: house_price_predictions.csv")
print("\nProject completed successfully 🎉")
