# Save this code in a new file: geospatial_analysis.py

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import numpy as np

# --- 1. Data Setup (Use the feature-rich data for context) ---
data_file = 'data/isro_missions_features_v2.csv'
if not os.path.exists(data_file):
    print(f"WARNING: Feature data '{data_file}' not found. Using a mockup DataFrame.")
    
    # Create a minimal DataFrame for visualization if the file is missing
    data = pd.DataFrame({
        'launch_site': ['Satish Dhawan Space Centre', 'Vikram Sarabhai Space Centre', 'Satish Dhawan Space Centre'],
        'launch_outcome': [1, 1, 0], # 1=Success, 0=Failure
        'launch_site_lat': [13.72, 8.52, 13.72],
        'launch_site_lon': [80.22, 76.94, 80.22],
        'launch_vehicle': ['PSLV', 'RH-200', 'GSLV Mk III']
    })
else:
    data = pd.read_csv(data_file)
    # Ensure lat/lon columns exist for visualization
    launch_site_coords = {
        'Satish Dhawan Space Centre': (13.72, 80.22),
        'Vikram Sarabhai Space Centre': (8.52, 76.94),
        'Thumba Launch Centre': (8.52, 76.94), # Merged with VSSC for simplicity
        'Chandipur Launch Site': (21.57, 87.06),
        'Sriharikota Range': (13.72, 80.22), # Merged with SDSC
    }
    data['launch_site_lat'] = data['launch_site'].map(lambda x: launch_site_coords.get(x, (np.nan, np.nan))[0])
    data['launch_site_lon'] = data['launch_site'].map(lambda x: launch_site_coords.get(x, (np.nan, np.nan))[1])
    data = data.dropna(subset=['launch_site_lat', 'launch_site_lon'])
    
print(f"Geospatial data loaded with {len(data)} entries.")

# --- 2. Map Initialization ---
# Center the map near India's launch sites
india_center = [15.0, 78.0]
m = folium.Map(location=india_center, zoom_start=5, tiles="CartoDB DarkMatter")

# --- 3. Add Launch Sites with Clustering ---
marker_cluster = MarkerCluster().add_to(m)

for index, row in data.iterrows():
    color = 'green' if row['launch_outcome'] == 1 else 'red'
    popup_text = (
        f"Site: {row['launch_site']}<br>"
        f"Vehicle: {row['launch_vehicle']}<br>"
        f"Outcome: {'Success' if row['launch_outcome'] == 1 else 'Failure'}"
    )
    
    folium.Marker(
        location=[row['launch_site_lat'], row['launch_site_lon']],
        popup=popup_text,
        icon=folium.Icon(color=color, icon='rocket', prefix='fa')
    ).add_to(marker_cluster)

# --- 4. Simulate Nominal Trajectory (PSLV/SSO from SDSC) ---
sdsc_lat, sdsc_lon = 13.72, 80.22

# Simulated ascent path for a polar/SSO launch (South-East path)
# Coordinates for the path (starting at SDSC)
trajectory_points = [
    [sdsc_lat, sdsc_lon],
    [10.0, 80.5],
    [5.0, 80.0],
    [0.0, 79.0],
    [-5.0, 78.0],
]

folium.PolyLine(
    trajectory_points,
    color='cyan',
    weight=2.5,
    opacity=0.8,
    popup='Simulated SSO Trajectory'
).add_to(m)

# --- 5. Define Safe Debris/Splashdown Zone (Bay of Bengal) ---
# Define a rectangular area in the Bay of Bengal for splashdown risk assessment
splashdown_zone = [
    [15.0, 85.0],
    [15.0, 95.0],
    [0.0, 95.0],
    [0.0, 85.0],
    [15.0, 85.0]
]

folium.Polygon(
    splashdown_zone,
    color='orange',
    fill=True,
    fill_color='orange',
    fill_opacity=0.2,
    popup='Nominal Debris/Splashdown Zone (Bay of Bengal)'
).add_to(m)

# --- 6. Save the Map ---
map_output_file = 'geospatial_launch_map.html'
m.save(map_output_file)

print("\n" + "="*60)
print(f"GEOSPATIAL ANALYSIS COMPLETE. Map saved to: {map_output_file}")
print("Rerun Streamlit to view the map in the 'Geospatial Analysis' tab.")
print("="*60)