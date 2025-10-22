import json

# --- RLV Financial Model Parameters (in millions of USD) ---
# Assuming 15 missions as the project lifecycle for analysis
N_MISSIONS = 15

# DISPOSABLE VEHICLE COST (e.g., PSLV/GSLV)
COST_PER_DISPOSABLE_LAUNCH = 90.0  # Cost of a typical disposable vehicle (per mission)

# REUSABLE VEHICLE COST (RLV)
RLV_RND_COST = 2000.0  # Initial Research & Development cost (fixed overhead)
RLV_REFURBISHMENT_COST = 10.0  # Cost to refurbish RLV between missions (per mission)
RLV_FABRICATION_COST = 500.0   # Initial cost to build the first RLV unit (fixed asset)

# --- Calculation ---

# 1. Total cost using disposable rockets over N_MISSIONS
total_disposable_cost = N_MISSIONS * COST_PER_DISPOSABLE_LAUNCH

# 2. Total cost using the Reusable Launch Vehicle (RLV)
total_reusable_cost = RLV_RND_COST + RLV_FABRICATION_COST + (N_MISSIONS * RLV_REFURBISHMENT_COST)

# 3. Financial Analysis
total_savings = total_disposable_cost - total_reusable_cost

# Break-Even Point (in missions): Number of flights needed for savings to offset R&D + Fabrication
# Savings per mission = COST_PER_DISPOSABLE_LAUNCH - RLV_REFURBISHMENT_COST
savings_per_mission = COST_PER_DISPOSABLE_LAUNCH - RLV_REFURBISHMENT_COST

# Missions to break even = (RLV R&D + RLV Fabrication) / Savings per mission
break_even_cost = RLV_RND_COST + RLV_FABRICATION_COST
break_even_point_missions = break_even_cost / savings_per_mission
break_even_point_missions = int(break_even_point_missions) + (1 if break_even_point_missions % 1 > 0 else 0) # Ceiling

# Strategic Insight
if total_savings > 0:
    roi_strategic_insight = f"The RLV program achieves significant cost savings after {break_even_point_missions} missions, projecting a total saving of ${total_savings:.2f}M over 15 missions. This justifies the initial heavy investment in R&D."
else:
    roi_strategic_insight = "The current cost model indicates that the RLV program does not achieve cost parity with disposable vehicles over 15 missions. Re-evaluation of refurbishment costs or mission frequency is required."

# --- Save Results ---
reusability_metrics = {
    'total_disposable_cost_M': round(total_disposable_cost, 2),
    'total_reusable_cost_M': round(total_reusable_cost, 2),
    'total_savings_M': round(total_savings, 2),
    'break_even_point_missions': break_even_point_missions,
    'roi_strategic_insight': roi_strategic_insight
}

with open('reusability_metrics.json', 'w') as f:
    json.dump(reusability_metrics, f, indent=4)

print(f"Reusability metrics saved to reusability_metrics.json. Break-Even: {break_even_point_missions} missions.")
