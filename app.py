import numpy as np 
import pandas as pd 
# import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model(r'C:\Users\krush\Desktop\Internship\FUTURE_ML_02\Dataset') 

# Streamlit UI
st.header('Stock Market Prediction')

stock = st.text_input('Enter Stock Symbol', 'B085BCWJV6')
start = '2012-01-01'
end = '2022-12-31'

# Download data
data = yf.download(stock, start, end)

# Check if data is fetched properly
if data.empty:
    st.error("Stock data not found. Please check the symbol and try again.")
else:
    st.subheader('Stock Data')
    st.write(data)

    # Prepare training and testing data
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    # If not enough data, stop the process
    if data_test.empty:
        st.warning("Not enough data in the test set to proceed.")
    else:
        scaler = MinMaxScaler(feature_range=(0,1))

        # Append last 100 days of training data to test set
        pas_100_days = data_train.tail(100)

        # Check if enough training data is present
        if pas_100_days.empty or len(data_test) < 1:
            st.warning("Not enough data to prepare test set.")
        else:
            data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

            # Scale test data safely
            if data_test.shape[0] >= 1:
                data_test_scale = scaler.fit_transform(data_test)

                # Prepare input sequences
                x = []
                y = []

                for i in range(100, data_test_scale.shape[0]):
                    x.append(data_test_scale[i-100:i])
                    y.append(data_test_scale[i, 0])

                if len(x) == 0:
                    st.warning("Not enough data to create sequences for prediction.")
                else:
                    x, y = np.array(x), np.array(y)

                    # Make predictions
                    y_predicted = model.predict(x)

                    st.success("Prediction completed! You can now plot or compare results.")
                    # (Add plotting code here if needed)

            else:
                st.warning("data_test is empty after combining. Cannot scale.")




# import numpy as np 
# import pandas as pd 
# import yfinance as yf
# import streamlit as st
# from tensorflow.keras.models import load_model  # ✅ Fixed this line
# from sklearn.preprocessing import MinMaxScaler

# # Load the model
# model = load_model(r'C:\Users\krush\Desktop\stock price prediction\Stock Prediction Model.keras') 

# # Streamlit UI
# st.header('Stock Market Prediction')

# stock = st.text_input('Enter Stock Symbol', 'B085BCWJV6')
# start = '2012-01-01'
# end = '2022-12-31'

# # Download data
# data = yf.download(stock, start, end)

# st.subheader('Stock Data')
# st.write(data)

# # Prepare training and testing data
# data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# scaler  = MinMaxScaler(feature_range=(0,1))

# # Append last 100 days of training data to test set
# pas_100_days = data_train.tail(100)
# data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# # Scale test data
# data_test_scale = scaler.fit_transform(data_test)

# x = []
# y = []

# for i in range(100, data_test_scale.shape[0]):
#     x.append(data_test_scale[i-100:i])
#     y.append(data_test_scale[i, 0])

# x, y = np.array(x), np.array(y)

# # You can now run predictions like:
# # y_predicted = model.predict(x)

# # Plotting and comparison can be added here as a next step
