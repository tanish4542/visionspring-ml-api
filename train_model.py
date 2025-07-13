import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv("visionspring_inventory_forecasting_data.csv")
df = df.drop(["Camp_ID", "Date"], axis=1)

# One-hot encode
df_encoded = pd.get_dummies(df, columns=["Location", "State", "Rural_Urban", "Season"])

# Separate features and target
X = df_encoded.drop("Frames_Used", axis=1)
y = df_encoded["Frames_Used"]

# Split
x_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "X_columns.pkl")

# Save encoded data as reference
df_encoded.to_csv("visionspring_inventory_forecasting_data_encoded.csv", index=False)

print("✅ model.pkl, X_columns.pkl, and encoded_reference.csv saved!")