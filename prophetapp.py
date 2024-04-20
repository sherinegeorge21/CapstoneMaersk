# First, import Streamlit and other necessary libraries
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define functions from your notebook with slight modifications for Streamlit
def prepare_data(df, country=None):
    if country and country.lower() != 'all':
        df = df[df['Location Activity Country Name'] == country]
    # Format these datetime objects to '01-11-2020' format
    

    df = df.rename(columns={'Activity Month Year': 'ds', 'Profit Loss USD': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['ds'] = df['ds'].dt.strftime('%d-%m-%Y')
    df = df.sort_values('ds')
    # Remove commas and strip spaces, then convert to numeric
    df['y'] = df['y'].str.replace(',', '').str.strip().astype(int)
    df = df.dropna(subset=['ds', 'y'])
    return df

def train_prophet(df):
    m = Prophet()
    m.fit(df)
    return m

def make_prediction(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def calculate_performance_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

# Define a function to encode the image for HTML embedding
# import base64
# def get_base64_encoded_image(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# # Define the layout
# st.markdown("""
#     <style>
#     .container {
#         display: flex;
#     }
#     .logo-text {
#         flex: 1;
#         display: flex;
#         align-items: center;  /* Aligns <h1> vertically */
#         justify-content: center;  /* Aligns <h1> horizontally */
#     }
#     .logo-img {
#         flex: 1;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#     }
#     img {
#         max-width: 100px;  /* Adjusts the size of the logo */
#         margin: 0;
#     }
#     h1 {
#         font-size: 24px; /* Adjust the size of your title font */
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Layout the logo and the title
# st.markdown("""
#     <div class="container">
#         <div class="logo-img">
#             <img src="data:image/png;base64,{}">
#         </div>
#         <div class="logo-text">
#             <h1>Maersk Profit and Loss Prediction</h1>
#         </div>
#     </div>
#     """.format(get_base64_encoded_image('maersk.png')), unsafe_allow_html=True)



# Load your data
@st.cache  # Use caching to load the data only once
def load_data(file_path):
    return pd.read_csv(file_path)

col1, col2 = st.columns([1, 6])  # Adjust the ratio if needed

with col1:
    st.image('maersk.png', width=120)  # You can adjust the width to fit your layout

with col2:
    st.title("Maersk Profit and Loss Prediction")

# Move file uploader to the sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Inputs for country and prediction periods are now in the sidebar
        country = st.text_input("Country to analyze (or 'all'):", 'all')
        periods = st.number_input("Months to predict into the future:", min_value=1, value=12)
        predict_button = st.button('Predict')

# Main area - show the forecasts here
if uploaded_file is not None and predict_button:
    filtered_df = prepare_data(df, country)
    if not filtered_df.empty:
        
        model = train_prophet(filtered_df)
        forecast = make_prediction(model, periods)
        
        actual_df = filtered_df # You need to implement this function to load actual data



        # Displaying the forecast
        st.write(forecast)
        
        # Plotting section
        median_value = forecast['yhat'].median()
        margin = forecast['yhat'].std() * 2.86

        lower_limit = median_value - margin
        upper_limit = median_value + margin

        fig, ax = plt.subplots()
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3, label='Confidence Interval')
        ax.set_title('Overall Forecasted Profit and Loss')
        ax.set_xlabel('Date')
        ax.set_ylabel('Profit/Loss USD')
        ax.set_ylim(lower_limit, upper_limit)
        st.pyplot(fig)

        # Second and Third plots here...
        # (No change needed, keep your code for the second and third plots here)
         # Second Plot - Bar Chart for 2024
        forecast['year'] = forecast['ds'].dt.year  # Extract year for filtering
        forecast_2024 = forecast[forecast['year'] == 2024]

        fig2, ax2 = plt.subplots()
        ax2.bar(forecast_2024['ds'], forecast_2024['yhat'], width=20, color='skyblue', label='Forecast')
        ax2.set_title('2024 Monthly Forecasted Profit and Loss')  # Title for the second plot
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Profit/Loss USD')
        ax2.legend()
        st.pyplot(fig2)

        # Third Plot - Pie Chart for Profit/Loss Distribution
        profit_2024 = forecast_2024[forecast_2024['yhat'] > 0]['yhat'].sum()
        loss_2024 = forecast_2024[forecast_2024['yhat'] < 0]['yhat'].sum()
        labels = ['Profit', 'Loss']
        sizes = [profit_2024, -loss_2024]  # Loss is negative, so we make it positive for the pie chart
        colors = ['lightgreen', 'salmon']

        fig3, ax3 = plt.subplots()
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax3.set_title('Profit and Loss Distribution for 2024')  # Title for the third plot
        st.pyplot(fig3)

                # Merge forecast and actual data on date
        # forecast = forecast.set_index('ds')
        # actual_df = actual_df.set_index('ds')
        # comparison_df = forecast.join(actual_df, how='inner', rsuffix='_actual')

        # # Calculate performance metrics
        # if not comparison_df.empty and 'y_actual' in comparison_df:
        #     mae, rmse, mape, r2 = calculate_performance_metrics(comparison_df['y_actual'], comparison_df['yhat'])
            
        #     # Display metrics
        #     st.metric("Mean Absolute Error (MAE)", mae)
        #     st.metric("Root Mean Squared Error (RMSE)", rmse)
        #     st.metric("Mean Absolute Percentage Error (MAPE)", mape)
        #     st.metric("R-squared", r2)


    else:
        st.error("No data available for the specified filter. Please try a different option.")
else:
    st.info('Awaiting the upload of a CSV file. Please use the sidebar to upload your file and input your parameters.')

       


