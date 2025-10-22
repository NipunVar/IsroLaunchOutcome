import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. DATA PREPARATION ---
# NOTE: Replace 'your_isro_data.csv' with the actual file path
# The data must have at least a date column and a numerical value column.
try:
    df = pd.read_csv('your_isro_data.csv')
except FileNotFoundError:
    print("Error: 'your_isro_data.csv' not found. Please update the file path.")
    # Create dummy data for demonstration if file is missing
    dates = pd.to_datetime(pd.date_range(start='2000-01-01', periods=100, freq='M'))
    values = [50 + i%12 * 5 + i/20 for i in range(100)] # Example trend + seasonality
    df = pd.DataFrame({'Date': dates, 'Outcome_Value': values})


# Rename columns to the strict Prophet format: 'ds' (datestamp) and 'y' (value)
# IMPORTANT: Adjust 'Date' and 'Outcome_Value' to match your actual column names
df = df.rename(columns={'Date': 'ds', 'Outcome_Value': 'y'})

# Ensure 'ds' is in datetime format
df['ds'] = pd.to_datetime(df['ds'])
# Ensure 'y' is numeric
df['y'] = pd.to_numeric(df['y'])


# --- 2. MODEL FITTING ---
# Instantiate the Prophet model
# You can customize it here (e.g., adding yearly_seasonality=False if your data is short)
model = Prophet(
    yearly_seasonality=True,  # Default: True, based on yearly cycles
    weekly_seasonality=False, # Assuming ISRO launches are not weekly seasonal
    daily_seasonality=False   # Assuming no daily pattern
)

# Fit the model to the data
model.fit(df)

# --- 3. FUTURE FORECASTING ---
# Create a DataFrame for future predictions.
# periods=12 means forecast 12 steps (months/days/years, based on your data frequency)
# freq='M' (Month), 'D' (Day), 'Y' (Year) - adjust this based on your data frequency
future = model.make_future_dataframe(periods=12, freq='M')

# Make the prediction
forecast = model.predict(future)

# --- 4. OUTPUT AND VISUALIZATION ---
print("\n--- Raw Forecast Results (Next 12 periods) ---")
# yhat: The predicted value
# yhat_lower/yhat_upper: The confidence interval
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# Plot the forecast (actuals in black dots, prediction line in blue)
fig1 = model.plot(forecast)
plt.title("Prophet Time Series Forecast")
plt.show()

# Plot the components (trend, seasonality, etc.)
fig2 = model.plot_components(forecast)
plt.show()