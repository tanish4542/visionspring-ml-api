import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess
df = pd.read_csv("visionspring_inventory_forecasting_data.csv")
df = df.drop(["Camp_ID", "Date"], axis=1)

categorical_cols = ["Location", "State", "Rural_Urban", "Season"]
df_encoded = pd.get_dummies(df, columns=categorical_cols)

X = df_encoded.drop("Frames_Used", axis=1)
y = df_encoded["Frames_Used"]

X_columns = X.columns  # Save for later use
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and metadata
joblib.dump(model, "model.pkl")
joblib.dump(X_columns, "X_columns.pkl")
df_encoded.to_csv("encoded_reference.csv", index=False)