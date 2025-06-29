import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('vegetable_madurai_2024.csv')

# Drop the 'date' column
df = df.drop(columns=['date'])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['vegetable', 'season', 'month', 'disaster', 'condition'], drop_first=True)

# Features and label
X = df.drop(['Price'], axis=1)
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, 'rf_model.pkl')
