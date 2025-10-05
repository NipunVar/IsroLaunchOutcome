import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 300

# ISRO vehicles and typical success rates
vehicles = ["SLV", "ASLV", "PSLV", "GSLV", "GSLV Mk III", "SSLV"]
vehicle_success_rate_map = {
    "SLV": 0.6,
    "ASLV": 0.64,
    "PSLV": 0.95,
    "GSLV": 0.85,
    "GSLV Mk III": 0.93,
    "SSLV": 0.75
}

orbit_types = ["LEO", "SSO", "GTO", "GEO", "MEO", "PO", "Sun-Synchronous", "Suborbital"]
mission_types = ["Communication", "Earth Observation", "Navigation", "Scientific", "Technology Demo", "Interplanetary"]
launch_windows = ["Morning", "Afternoon", "Evening", "Night"]

# Random assignment
launch_vehicle = np.random.choice(vehicles, size=n)
orbit_type = np.random.choice(orbit_types, size=n)
payload_weight_kg = np.random.uniform(50, 4500, n).round()
temperature_C = np.random.normal(25, 10, n).clip(-10, 45).round(1)
wind_speed_kmh = np.random.normal(10, 8, n).clip(0, 50).round(1)
humidity_percent = np.random.uniform(10, 90, n).round(1)
launch_window = np.random.choice(launch_windows, size=n)
mission_type = np.random.choice(mission_types, size=n)
system_health_index = np.random.beta(5, 2, n).round(3)
vehicle_success_rate = [vehicle_success_rate_map[v] for v in launch_vehicle]

# Realistic date range 2000-2025
launch_dates = pd.to_datetime(
    np.random.randint(
        pd.Timestamp('2000-01-01').value // 10**9,
        pd.Timestamp('2025-01-01').value // 10**9,
        n
    ), unit='s'
)

# Simulate realistic outcomes
def simulate_outcome(system_health, success_rate):
    prob_success = system_health * success_rate
    return np.random.binomial(1, prob_success)

launch_outcome = [simulate_outcome(sh, sr) for sh, sr in zip(system_health_index, vehicle_success_rate)]

# Create DataFrame
data = pd.DataFrame({
    "mission_id": [f"ISRO-{i+1:03d}" for i in range(n)],
    "launch_date": launch_dates,
    "launch_vehicle": launch_vehicle,
    "orbit_type": orbit_type,
    "payload_weight_kg": payload_weight_kg,
    "temperature_C": temperature_C,
    "wind_speed_kmh": wind_speed_kmh,
    "humidity_percent": humidity_percent,
    "launch_window": launch_window,
    "mission_type": mission_type,
    "system_health_index": system_health_index,
    "vehicle_success_rate": vehicle_success_rate,
    "launch_outcome": launch_outcome
})

# Add launch site info (dummy random assignment)
launch_sites = [
    ('Satish Dhawan Space Centre', 13.72, 80.15),
    ('Vikram Sarabhai Space Centre', 8.52, 76.94),
    ('Thumba Launch Centre', 8.52, 76.94),
    ('Chandipur Launch Site', 21.57, 87.06),
    ('Sriharikota Range', 13.70, 80.23),
]

assigned_sites = np.random.choice(len(launch_sites), size=len(data))

site_names = [launch_sites[i][0] for i in assigned_sites]
site_latitudes = [launch_sites[i][1] for i in assigned_sites]
site_longitudes = [launch_sites[i][2] for i in assigned_sites]

data['launch_site'] = site_names
data['launch_site_lat'] = site_latitudes
data['launch_site_lon'] = site_longitudes

# Save dataset
os.makedirs("data", exist_ok=True)
data.to_csv("data/isro_300_missions.csv", index=False)
print("Synthetic ISRO launches data with launch sites saved as data/isro_300_missions.csv")
