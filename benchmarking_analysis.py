# Save this code in a file: benchmarking_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Step 23: Define Simulated Benchmarking Data ---
BENCHMARK_DATA = {
    'Agency': ['ISRO', 'SpaceX', 'NASA (Commercial)', 'Roscosmos', 'CNSA'],
    'Success Rate': [None, 0.98, 0.96, 0.94, 0.95], 
    'Cost per kg to LEO (USD)': [None, 2700, 10000, 4500, 3500],
    'Total Launches (Simulated)': [None, 350, 150, 500, 400]
}
benchmark_df = pd.DataFrame(BENCHMARK_DATA).set_index('Agency')

# --- Step 24: Calculate ISRO's Metrics ---
def calculate_isro_metrics(data_file='data/isro_300_missions.csv'):
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Ensure it is in the 'data' folder.")
        return None
    
    df = pd.read_csv(data_file)
    
    # 1. Success Rate
    isro_success_rate = df['launch_outcome'].mean() 
    
    # 2. Cost per kg to LEO (Simplified calculation)
    avg_vehicle_cost_M = 21.5 
    low_cost_vehicles = df[df['launch_vehicle'].isin(['PSLV', 'SSLV'])]
    avg_payload_kg = low_cost_vehicles['payload_weight_kg'].mean()
    
    isro_cost_per_kg = (avg_vehicle_cost_M * 1_000_000) / avg_payload_kg
    
    return isro_success_rate, isro_cost_per_kg, len(df)

# --- Step 25: Visualize Competitiveness ---
def visualize_benchmarks(isro_rate, isro_cost, isro_launches):
    # Populate the ISRO row
    benchmark_df.loc['ISRO', 'Success Rate'] = isro_rate
    benchmark_df.loc['ISRO', 'Cost per kg to LEO (USD)'] = isro_cost
    benchmark_df.loc['ISRO', 'Total Launches (Simulated)'] = isro_launches
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0a0a23')
    
    # Subplot 1: Success Rate
    rates = benchmark_df['Success Rate']
    rates.sort_values(ascending=False).plot(kind='barh', ax=axes[0], color=['yellow' if idx == 'ISRO' else 'cyan' for idx in rates.index])
    axes[0].set_title('Launch Success Rate Comparison', color='#dcdcdc', fontsize=14)
    axes[0].set_xlabel('Success Rate (%)', color='#dcdcdc')
    axes[0].set_ylabel('Agency', color='#dcdcdc')
    axes[0].tick_params(axis='x', colors='#dcdcdc')
    axes[0].tick_params(axis='y', colors='#dcdcdc')
    axes[0].set_facecolor('#0a0a23')

    # Subplot 2: Cost per kg to LEO
    costs = benchmark_df['Cost per kg to LEO (USD)']
    costs.sort_values(ascending=False).plot(kind='barh', ax=axes[1], color=['yellow' if idx == 'ISRO' else 'cyan' for idx in costs.index])
    axes[1].set_title('Estimated Cost per kg to LEO (USD)', color='#dcdcdc', fontsize=14)
    axes[1].set_xlabel('Cost per kg (USD)', color='#dcdcdc')
    axes[1].set_ylabel('', color='#dcdcdc') 
    axes[1].tick_params(axis='x', colors='#dcdcdc')
    axes[1].tick_params(axis='y', colors='#dcdcdc')
    axes[1].set_facecolor('#0a0a23')
    
    plt.tight_layout()
    plt.gcf().set_facecolor('#0a0a23')
    
    # CRITICAL FIX: Save the figure here!
    plt.savefig('benchmarking_chart.png', facecolor=fig.get_facecolor()) 
    print("\nChart saved successfully as 'benchmarking_chart.png'")
    plt.show() # Keep if you want the pop-up window

# --- Main Execution ---
if __name__ == "__main__":
    metrics = calculate_isro_metrics()
    if metrics:
        isro_rate, isro_cost, isro_launches = metrics
        visualize_benchmarks(isro_rate, isro_cost, isro_launches)