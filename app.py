import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

st.title("📈 Stock Market Prediction (Custom Dataset + Symbol Dropdown)")

# Check if model file exists and load with error handling
model_path = "stockpriceprediction.keras"

try:
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load stock symbols metadata
meta_df = pd.read_csv("symbols_valid_meta.csv")

# Filter only traded symbols
valid_symbols = meta_df[meta_df["Nasdaq Traded"] == "Y"]

# Create dropdown menu
symbols_list = valid_symbols["Symbol"].dropna().unique()
selected_symbol = st.selectbox("Select a stock symbol", sorted(symbols_list))

# Try to load CSV for the selected symbol
file_path = f"datasets/{selected_symbol}.csv"
if not os.path.exists(file_path):
    st.warning(f"No dataset found for symbol: {selected_symbol}. Please make sure {selected_symbol}.csv exists in the datasets folder.")
    st.stop()

# Load stock data
df = pd.read_csv(file_path)

# Check for Date and Close columns
if 'Date' not in df.columns or 'Close' not in df.columns:
    st.error("CSV must have 'Date' and 'Close' columns.")
    st.stop()

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Optional Date Filter
st.subheader("📅 Filter by Date Range (optional)")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Filter data
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
st.write(f"Showing data from {start_date} to {end_date} ({len(filtered_df)} rows)")

# Show preview
st.dataframe(filtered_df[['Date', 'Close']].head())

# Check data length
if len(filtered_df) < 150:
    st.warning("Please select a longer date range. Need at least 150 rows after filtering.")
    st.stop()

# Use Close prices
close_prices = filtered_df['Close'].reset_index(drop=True)

# Train-test split
train_size = int(len(close_prices) * 0.80)
data_train = close_prices[:train_size]
data_test = close_prices[train_size:]

# Add last 100 days to test
past_100_days = data_train[-100:]
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_test_scaled = scaler.fit_transform(final_test_data.values.reshape(-1, 1))

# Prepare input
x_test, y_test = [], []
for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Inverse scale
scale_factor = 1 / scaler.scale_[0]
y_predicted_actual = y_predicted * scale_factor
y_test_actual = y_test * scale_factor

# Plot
st.subheader("📊 Actual vs Predicted Closing Prices")
fig = plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_predicted_actual, label='Predicted Price')
plt.legend()
plt.xlabel("Days")
plt.ylabel("Stock Price")
st.pyplot(fig)
