import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from utils import nifty50_companies

# --- Load Stock Data ---
def load_data(company):
    filename = f"data/{company}.csv"
    data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    return data

# --- Load Pretrained LSTM Model ---
def load_model_file(company):
    model_filename = f"models/{company}_model.keras"
    model = load_model(model_filename)
    return model

# --- Make Predictions ---
def make_predictions(model, data, num_predictions, scaler):
    test_data = data[-100:].reshape(1, -1)
    predictions = []
    for _ in range(num_predictions):
        predicted_value = model.predict(test_data)[0, 0]
        predictions.append(predicted_value)
        test_data = np.roll(test_data, -1)
        test_data[0, -1] = predicted_value
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# --- Convert DataFrame to CSV for Download ---
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# --- Apply Custom Styles ---
def set_custom_styles():
    st.markdown("""
        <style>
            body {
                background-color: #e3f2fd !important;
                font-family: 'Segoe UI', sans-serif;
            }
            .main {
                background-color: #e3f2fd !important;
                padding: 1rem;
                font-family: 'Segoe UI', sans-serif;
            }
            h1, h2, h3, .stMarkdown, .stDataFrame {
                animation: fadeIn 1.2s ease-in-out;
            }
            @keyframes fadeIn {
                0% { opacity: 0; transform: translateY(-10px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            button[kind="primary"] {
                animation: pulse 1.8s infinite;
                border-radius: 10px;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.6); }
                70% { box-shadow: 0 0 0 10px rgba(0, 123, 255, 0); }
                100% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }
            }
            .stDownloadButton, .stButton>button {
                transition: all 0.3s ease-in-out;
                border-radius: 10px;
            }
            .stDownloadButton:hover, .stButton>button:hover {
                transform: scale(1.05);
                background-color: #64b5f6 !important;
                color: white;
            }
            .stSidebar {
                background-color: #bbdefb !important;
                padding: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Main App ---
def main():
    st.set_page_config(page_title="Stock Forecasting App", layout="wide")
    set_custom_styles()

    # Basic Animation using CSS (No Lottie)
    st.markdown("<h1 style='animation: fadeIn 2s;'>📊 AI-Based Stock Price Forecasting (NIFTY 50)</h1>", unsafe_allow_html=True)
    st.markdown("Predict future stock prices using advanced LSTM neural networks and visualize trends interactively.")

    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        company = st.selectbox("📌 Select Company", list(nifty50_companies.keys()))
        if st.button("📥 Load Data"):
            data = load_data(nifty50_companies[company])
            st.session_state.data = data
            st.session_state.company = company

    if "data" in st.session_state:
        st.subheader(f"🗂 {st.session_state.company} | Last 5 Stock Records")
        st.dataframe(st.session_state.data.tail(5), use_container_width=True)

        st.markdown("## 📈 Historical Stock Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Open'], mode='lines', name='Open'))
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['High'], mode='lines', name='High'))
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Low'], mode='lines', name='Low'))
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price (INR)', height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            num_predictions = st.number_input("🔮 Days to Predict", min_value=1, max_value=90, value=60)

        with col2:
            if st.button("🚀 Start Forecasting"):
                with st.spinner("Running AI Model..."):
                    model = load_model_file(nifty50_companies[st.session_state.company])
                    scaler = MinMaxScaler()

                    st.session_state.data['Close'] = pd.to_numeric(st.session_state.data['Close'], errors='coerce')
                    st.session_state.data.dropna(subset=['Close'], inplace=True)

                    if st.session_state.data['Close'].shape[0] < 100:
                        st.error("⚠️ Not enough data. At least 100 valid rows are required.")
                        return

                    scaled_data = scaler.fit_transform(st.session_state.data['Close'].values.reshape(-1, 1))
                    predictions = make_predictions(model, scaled_data, num_predictions, scaler)

                    st.session_state.predictions = predictions
                    st.session_state.num_predictions = num_predictions

    if "predictions" in st.session_state:
        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=st.session_state.data.index[-1] + pd.Timedelta(days=1), periods=st.session_state.num_predictions, freq='D'),
            "Forecasted_value": st.session_state.predictions
        })
        forecast_df["Forecasted_value"] = forecast_df["Forecasted_value"].astype("float64").round(2)

        st.markdown(f"## 📉 Forecasted Prices: {st.session_state.company}")
        st.dataframe(forecast_df, use_container_width=True)

        st.markdown("## 📊 Forecast vs Historical Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Original Close'))
        fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted_value"], mode='lines', name='Forecasted Close'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price (INR)', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📦 Download Forecast & Historical Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("⬇️ Download Original", convert_df_to_csv(st.session_state.data), file_name="original_data.csv")
        with col2:
            st.download_button("⬇️ Download Forecast", convert_df_to_csv(forecast_df), file_name="forecast_data.csv")
        with col3:
            if st.button("🔄 Reset App"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()

# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from utils import nifty50_companies
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# def load_data(company):
#     filename = f"data/{company}.csv"
#     data = pd.read_csv(filename, index_col='Date', parse_dates=True)
#     return data

# def load_model_file(company):
#     model_filename = f"models/{company}_model.keras"
#     model = load_model(model_filename)
#     return model

# def make_predictions(model, data, num_predictions, scaler):
#     test_data = data[-100:].reshape(1, -1)
#     predictions = []
#     for _ in range(num_predictions):
#         predicted_value = model.predict(test_data)
#         predictions.append(predicted_value[0, 0])
#         test_data = np.roll(test_data, -1)
#         test_data[0, -1] = predicted_value
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#     return predictions.flatten()

# def convert_df_to_csv(df):
#     return df.to_csv(index=True).encode('utf-8')

# def main():
#     st.title("📈 Stock Price Forecasting App")

#     company = st.selectbox("Select Company", list(nifty50_companies.keys()))

#     if st.button("Get Data"):
#         data = load_data(nifty50_companies[company])
#         st.session_state.data = data
#         st.session_state.company = company

#     if "data" in st.session_state:
#         st.subheader(f"{st.session_state.company} Original Data: Last 5 Records")
#         st.table(st.session_state.data.tail(5))

#         st.subheader(f"{st.session_state.company} Complete Data Visualization")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Open'], mode='lines', name='Open'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['High'], mode='lines', name='High'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Low'], mode='lines', name='Low'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close'))
#         fig.update_layout(xaxis_title='Date', yaxis_title='Price')
#         st.plotly_chart(fig)

#         num_predictions = st.number_input("No of Next Predictions (1–90 days):", min_value=1, max_value=90, value=60)

#         if st.button("Forecast"):
#             model = load_model_file(nifty50_companies[st.session_state.company])
#             scaler = MinMaxScaler(feature_range=(0, 1))

#             # ✅ Clean the 'Close' column
#             st.session_state.data['Close'] = pd.to_numeric(st.session_state.data['Close'], errors='coerce')
#             st.session_state.data.dropna(subset=['Close'], inplace=True)

#             if st.session_state.data['Close'].shape[0] < 100:
#                 st.error("Not enough data after cleaning. At least 100 valid data points are required.")
#                 return

#             scaled_data = scaler.fit_transform(st.session_state.data['Close'].values.reshape(-1, 1))
#             predictions = make_predictions(model, scaled_data, num_predictions, scaler)
#             st.session_state.predictions = predictions
#             st.session_state.num_predictions = num_predictions

#     if "predictions" in st.session_state:
#         forecast_df = pd.DataFrame({
#             "Date": pd.date_range(start=st.session_state.data.index[-1] + pd.Timedelta(days=1), periods=st.session_state.num_predictions, freq='D'),
#             "Forecasted_value": st.session_state.predictions
#         })
#         forecast_df["Forecasted_value"] = forecast_df["Forecasted_value"].astype("float64").round(2)

#         st.subheader(f"📊 Forecasted Values for {st.session_state.company}")
#         st.table(forecast_df)

#         st.subheader(f"📈 {st.session_state.company} Forecasting")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Original Data'))
#         forecast_index = pd.date_range(start=st.session_state.data.index[-1] + pd.Timedelta(days=1), periods=st.session_state.num_predictions, freq='D')
#         fig.add_trace(go.Scatter(x=forecast_index, y=st.session_state.predictions, mode='lines', name='Forecasted Data'))
#         fig.update_layout(xaxis_title='Date', yaxis_title='Price')
#         st.plotly_chart(fig)

#         st.subheader("⬇️ Download Data and Results")
#         historical_csv = convert_df_to_csv(st.session_state.data)
#         forecast_csv = convert_df_to_csv(forecast_df)

#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.download_button(label="Download Original Data",
#                                data=historical_csv,
#                                file_name=f"{nifty50_companies[st.session_state.company]}_historical_data.csv",
#                                mime='text/csv')

#         with col2:
#             st.download_button(label="Download Forecasted Results",
#                                data=forecast_csv,
#                                file_name=f"{nifty50_companies[st.session_state.company]}_forecast_results.csv",
#                                mime='text/csv')

#         with col3:
#             if st.button("Reset"):
#                 st.session_state.clear()
#                 st.rerun()

# if __name__ == "__main__":
#     main()


# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from utils import nifty50_companies
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# def load_data(company):
#     filename = f"data/{company}.csv"
#     data = pd.read_csv(filename, index_col='Date', parse_dates=True)
#     return data

# def load_model_file(company):
#     model_filename = f"models/{company}_model.keras"
#     model = load_model(model_filename)
#     return model

# def make_predictions(model, data, num_predictions, scaler):
#     test_data = data[-100:].reshape(1, -1)
#     predictions = []
#     for _ in range(num_predictions):
#         predicted_value = model.predict(test_data)
#         predictions.append(predicted_value[0, 0])
#         test_data = np.roll(test_data, -1)
#         test_data[0, -1] = predicted_value
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#     return predictions.flatten()

# def convert_df_to_csv(df):
#     return df.to_csv(index=True).encode('utf-8')

# def main():
#     st.title("Stock Price Forecasting App")

#     company = st.selectbox("Select Company", list(nifty50_companies.keys()))

#     if st.button("Get Data"):
#         data = load_data(nifty50_companies[company])
#         st.session_state.data = data
#         st.session_state.company = company

#     if "data" in st.session_state:
#         st.subheader(f"{st.session_state.company} Original Data: Last 5 Records")
#         st.table(st.session_state.data.tail(5))

#         st.subheader(f"{st.session_state.company} Complete Data Visualization")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Open'], mode='lines', name='Open'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['High'], mode='lines', name='High'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Low'], mode='lines', name='Low'))
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Close'))
#         fig.update_layout(xaxis_title='Date', yaxis_title='Price')
#         st.plotly_chart(fig)

#         num_predictions = st.number_input("No of Next Predictions(1-90 days):", 
#                                           min_value=1, max_value=90, value=60)

#         if st.button("Forecast"):
#             model = load_model_file(nifty50_companies[st.session_state.company])
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             scaled_data = scaler.fit_transform(st.session_state.data['Close'].values.reshape(-1, 1))
#             predictions = make_predictions(model, scaled_data, num_predictions, scaler)
#             st.session_state.predictions = predictions
#             st.session_state.num_predictions = num_predictions

#     if "predictions" in st.session_state:
#         forecast_df = pd.DataFrame({
#             "Date": pd.date_range(start=st.session_state.data.index[-1] + pd.Timedelta(days=1), periods=st.session_state.num_predictions, freq='D'),
#             "Forecasted_value": st.session_state.predictions
#         })
#         forecast_df["Forecasted_value"] = forecast_df["Forecasted_value"].astype("float64").round(2)
#         st.subheader(f"Forecasted Values for {st.session_state.company}")
#         st.table(forecast_df)

#         st.subheader(f"{st.session_state.company} Forecasting")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data['Close'], mode='lines', name='Original Data'))
#         forecast_index = pd.date_range(start=st.session_state.data.index[-1] + pd.Timedelta(days=1), periods=st.session_state.num_predictions, freq='D')
#         fig.add_trace(go.Scatter(x=forecast_index, y=st.session_state.predictions, mode='lines', name='Forecasted Data'))
#         fig.update_layout(xaxis_title='Date', yaxis_title='Price')
#         st.plotly_chart(fig)

#         st.subheader("Download Data and Results")
#         historical_csv = convert_df_to_csv(st.session_state.data)
#         forecast_csv = convert_df_to_csv(forecast_df)

#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.download_button(label="Download Original Data",
#                                data=historical_csv,
#                                file_name=f"{nifty50_companies[st.session_state.company]}_historical_data.csv",
#                                mime='text/csv')
        
#         with col2:
#             st.download_button(label="Download Forecasted Results",
#                                data=forecast_csv,
#                                file_name=f"{nifty50_companies[st.session_state.company]}_forecast_results.csv",
#                                mime='text/csv')
#         with col3:
#             if st.button("Reset"):
#                 st.session_state.clear()
#                 st.rerun()

# if __name__ == "__main__":
#     main()