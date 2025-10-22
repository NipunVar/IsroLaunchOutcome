import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
# Import models for Stacking
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# Load feature-rich dataset (assuming you ran advanced_feature_engineering.py)
data_file = 'data/isro_missions_features_v2.csv'
if not os.path.exists(data_file):
    # Fallback/Warning if v2 is not found
    print(f"WARNING: Feature-rich data file '{data_file}' not found. Using original data.")
    data_file = 'data/isro_300_missions.csv' 

if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found. Run generate_isro_data.py and advanced_feature_engineering.py.")

data = pd.read_csv(data_file)
print(f"Data loaded with shape: {data.shape}")

# Define features (X) and target (y)
X = data.drop('launch_outcome', axis=1)
y = data['launch_outcome']

# Define feature groups, ensuring the new features are included
categorical_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site', 'orbit_type']
numeric_features = [
    'payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 
    'system_health_index', 'vehicle_success_rate', 
    # NEW FEATURES from Project 3 (Phase 1)
    'launch_month', 'launch_quarter', 'mission_complexity_score', 'launch_window_risk_index'
]
# Filter features to only those present in the dataframe
categorical_features = [f for f in categorical_features if f in X.columns]
numeric_features = [f for f in numeric_features if f in X.columns]

# --- Define Preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        # Apply scaling to numeric features (important for SVM and Logistic Regression in the stack)
        ('num', StandardScaler(), numeric_features),
        # Apply One-Hot Encoding
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# --- Define Stacking Classifier ---
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    # SVC requires scaling, which is handled in the preprocessor
    ('svc', SVC(kernel='linear', probability=True, random_state=42)),
    ('lr', LogisticRegression(solver='liblinear', random_state=42)),
]

# Define the Stacking Classifier (Final Estimator is XGBoost)
stacking_classifier = StackingClassifier(
    estimators=estimators, 
    final_estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    cv=5,  # 5-fold cross-validation
    n_jobs=-1
)

# --- Update and Train Pipeline ---
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_classifier)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*50)
print("TRAINING STACKING CLASSIFIER WITH ADVANCED FEATURES")
print("="*50)

# Fit the pipeline
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nStacking Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model with the CORRECT name for the API
joblib.dump(model, 'isro_launch_model_v2.pkl')
print("\nModel saved as 'isro_launch_model_v2.pkl'. READY FOR API DEPLOYMENT.")

# The old plot logic is removed for simplicity, but you can uncomment it if needed.
# plt.show()