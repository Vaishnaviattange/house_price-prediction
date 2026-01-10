import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("=== Training House Price Model ===")

# Load your cleaned dataset
df = pd.read_csv("Cleaned train.csv")

# Input features (same as UI)
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "LotArea"
]

# ✅ FIXED target column name
target = "Saleprice"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully 🎉")
