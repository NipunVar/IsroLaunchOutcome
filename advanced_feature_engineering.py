# Save this code in a new file: advanced_feature_engineering.py

import pandas as pd
import numpy as np
import os

data_file = 'data/isro_300_missions.csv'

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Run generate_isro_data.py first.")
else:
    data = pd.read_csv(data_file)
    data['launch_date'] = pd.to_datetime(data['launch_date'])
    print("Original Data Info:", data.shape)
    
    # ==========================================================
    # Phase 1: Advanced Feature Engineering (Steps 14-16)
    # ==========================================================
    
    # --- Step 14: Time-of-Year Features ---
    data['launch_month'] = data['launch_date'].dt.month
    data['launch_quarter'] = data['launch_date'].dt.quarter
    
    # --- Step 15: Mission Complexity Score ---
    
    # Assign higher scores to more complex missions/orbits
    orbit_complexity_map = {
        'Suborbital': 1, 
        'LEO': 2, 'SSO': 3, 'PO': 3, 
        'MEO': 4, 
        'GTO': 5, 'GEO': 6, # High energy/complex orbit insertion
        'Interplanetary': 7 
    }
    
    mission_type_complexity_map = {
        'Technology Demo': 1, 'Navigation': 2, 
        'Earth Observation': 3, 'Communication': 4, 
        'Scientific': 5, 'Interplanetary': 6 
    }
    
    # Ensure map keys cover all unique values by handling potential missing keys
    data['orbit_score'] = data['orbit_type'].map(orbit_complexity_map).fillna(0)
    data['mission_score'] = data['mission_type'].map(mission_type_complexity_map).fillna(0)
    
    # Total Complexity Score
    data['mission_complexity_score'] = data['orbit_score'] + data['mission_score']
    
    # --- Step 16: Launch Slot Analysis (Simplified Risk Index) ---
    
    # Assign a risk score based on the launch window
    # Morning/Afternoon are usually standard, Night/Evening might carry more risk (operational fatigue, thermal changes)
    launch_window_risk_map = {
        'Morning': 1.0, 
        'Afternoon': 1.1, 
        'Evening': 1.5, 
        'Night': 1.3 
    }
    data['launch_window_risk_index'] = data['launch_window'].map(launch_window_risk_map).fillna(1.0)
    
    # Clean up intermediary columns if desired
    data.drop(columns=['orbit_score', 'mission_score'], inplace=True)
    
    print("\n" + "="*50)
    print("ADVANCED FEATURE ENGINEERING COMPLETE")
    print("New Features Added:")
    print(data[['launch_date', 'launch_month', 'mission_complexity_score', 'launch_window_risk_index']].head())
    print("New Data Info:", data.shape)
    
    # Save the feature-rich data (optional, but useful)
    data.to_csv('data/isro_missions_features_v2.csv', index=False)