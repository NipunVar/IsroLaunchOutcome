import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.inspection import PartialDependenceDisplay
import joblib

# Load dataset
data_file = 'data/isro_300_missions.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found. Run generate_isro_data.py to create it.")
data = pd.read_csv(data_file)

# Confirm data
print(data.info())
print(data['launch_outcome'].value_counts())

# Map launch_site to lat/lon coordinates
launch_site_coords = {
    'Satish Dhawan Space Centre': (13.72, 80.15),
    'Vikram Sarabhai Space Centre': (8.52, 76.94),
    'Thumba Launch Centre': (8.52, 76.94),
    'Chandipur Launch Site': (21.57, 87.06),
    'Sriharikota Range': (13.70, 80.23),
}

data['launch_site_lat'] = data['launch_site'].map(lambda x: launch_site_coords.get(x, (np.nan, np.nan))[0])
data['launch_site_lon'] = data['launch_site'].map(lambda x: launch_site_coords.get(x, (np.nan, np.nan))[1])

# Drop rows with missing crucial data (optional)
data = data.dropna(subset=['launch_site_lat', 'launch_site_lon'])

# Prepare features and target
X = data.drop(columns=['launch_outcome', 'mission_id', 'launch_date', 'orbit_type', 'launch_site_lat', 'launch_site_lon'])
y = data['launch_outcome']

# Categorical features now include 'launch_site'
categorical_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessor on training data and get feature names
preprocessor.fit(X_train)
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = list(ohe.get_feature_names_out(categorical_features))

num_cols = [col for col in X_train.columns if col not in categorical_features]
all_feature_names = cat_feature_names + num_cols

# Define and train pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Partial dependence plots
features_to_plot = ['system_health_index', 'payload_weight_kg', 'temperature_C']
disp = PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot)
disp.figure_.set_size_inches(10, 6)
plt.tight_layout()
plt.show()

# Save the pipeline
joblib.dump(model, 'launch_model_pipeline.pkl')
print("Model pipeline saved as launch_model_pipeline.pkl")
