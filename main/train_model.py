import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
df = pd.read_csv('parkinsons_features.csv')

# 2. Separate features (X) and the target variable (y)
X = df.drop('status', axis=1)
y = df['status']

# --- THE FIX: IMPUTATION ---
# Instead of dropping rows with missing values, we fill them.
print("--- Data Cleaning ---")
print(f"Features have {X.isnull().sum().sum()} missing values before imputation.")
X.fillna(X.mean(), inplace=True) # Fill NaNs with the mean of their respective columns
print("Missing values have been filled using the mean (imputation).")
print("-" * 25)


# 3. Split the data into training and testing sets
print("Splitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split successfully.")

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Random Forest Classifier
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# 6. Save the trained model and the scaler
joblib.dump(model, 'parkinsons_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler have been saved successfully!")

# --- 7. Evaluate the model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinsons']))