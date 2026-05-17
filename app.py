import streamlit as st 
import joblib
import pandas as pd 
import numpy as np 
from xgboost import XGBRegressor


st.set_page_config(page_title="⚡ Forecasting", layout="wide")

st.title("Electricity Consumption Forecasting Dashboard")


st.sidebar.header("About")

st.sidebar.write("""
The dashboard uses machine learning to forecast electricity consumption for the upcoming week based
on historical data
""")
st.sidebar.subheader("Model Details")
st.sidebar.write("""
- Algorithm : XGBoost
- Features : Lag + Rolling Features
""")
st.sidebar.subheader("Accuracy Metrics")
st.sidebar.write("""
- MAE : 0.54 kW
- R² Score : 0.99
- MAPE : 3.39%
""")
st.sidebar.subheader("Developer")
st.sidebar.write("Name : Anzish Gul")
st.sidebar.write("Email : [anzishgul2@gmail.com](mailto:anzishgul2@gmail.com)")
st.sidebar.write("GitHub : [AnzishGul2](https://github.com/AnzishGul2)")
st.sidebar.write("LinkedIn : [anzish-gul-5355aa40b](https://www.linkedin.com/in/anzish-gul-5355aa40b)")

import os
model = XGBRegressor()

# Check if file exists (ab root mein hai)
model_path = "xgb_model.json"
if os.path.exists(model_path):
    model.load_model(model_path)
else:
    st.error("Model file not found! Please check GitHub repository.")
    st.stop()

features = joblib.load("features.pkl")
all_columns = joblib.load("all_columns.pkl")

st.write("This dashboard will show you the total consumption of next 7 days")


uploaded_file = st.file_uploader(
    "Upload data of last 14 days",
    type=["csv", "xlsx"]
)


with st.expander(" View Required Data Format (Click to expand)"):
    st.write("**Option 1: Add only 2 columns (Minimum):**")
    st.write("""
    - Date[Y-m-D]
    - Global_active_power
    """)
    st.write("**All other columns will be filled with 0**")
    st.write("")
    st.write("**Option 2: Add all columns (Recommended for better accuracy):**")
    st.write("""
    - Date[Y-m-D]
    - Global_active_power
    - Global_reactive_power
    - Voltage
    - Global_intensity
    - Sub_metering_1
    - Sub_metering_2
    - Sub_metering_3
    """)

    

def create_features(df):

    df = df.sort_index()

    cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]

    windows = [3, 7]


    for col in cols:
        for w in windows:
            df[f'{col}_roll_mean_{w}'] = df[col].rolling(w).mean()
            df[f'{col}_roll_std_{w}'] = df[col].rolling(w).std()


    for col in cols:
        for l in [1, 2, 3, 7]:
            df[f'{col}_lag_{l}'] = df[col].shift(l)

  
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df



if uploaded_file is not None:


    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

  
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    all_required_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
    
  
    missing_cols_list = []
    for col in all_required_cols:
        if col not in df.columns:
            df[col] = 0
            missing_cols_list.append(col)
    
 
    if 'Global_active_power' not in df.columns:
        st.error(" 'Global_active_power' column is required!")
        st.stop()
    
  
    if missing_cols_list:
        st.warning(f" Missing columns: {missing_cols_list}. Filled with 0.")
        st.info(" For better accuracy, please provide all columns.")
    

    st.write("### Uploaded Data Preview")
    st.dataframe(df.head(5))

    st.success("File uploaded successfully")

    df_features = create_features(df)


    X = df_features.copy()


    X = X.reindex(columns=features, fill_value=0)

 
    with st.spinner("Generating prediction..."):
        preds = model.predict(X)
        

        preds = preds / 7
        total_prediction = np.sum(preds)

 
    st.success("Forecast generated successfully")

    st.metric(
        label=" Next Week Total Consumption",
        value=f"{total_prediction:.2f} kW",
        delta=f"~{total_prediction/7:.2f} kW per day"
    )
    
    st.markdown("---")
    
 
    st.subheader(" Daily Consumption Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days = [f"Day {i+1}" for i in range(len(preds))]
        chart_df = pd.DataFrame({
            "Day": days,
            "Predicted (kW)": preds.round(2)
        })
        st.bar_chart(chart_df.set_index("Day"), use_container_width=True)
    
    with col2:
        chart_df_line = pd.DataFrame({
            "Day": range(1, len(preds) + 1),
            "kW": preds.round(2)
        })
        st.line_chart(chart_df_line.set_index("Day"), use_container_width=True)
    

    st.subheader(" Cumulative Consumption")
    cumulative = np.cumsum(preds)
    cum_df = pd.DataFrame({
        "Day": range(1, len(preds) + 1),
        "Cumulative (kW)": cumulative.round(2)
    })
    st.area_chart(cum_df.set_index("Day"), use_container_width=True)
    
  
    st.subheader(" Last Week Consumption Pattern")
    if len(df) >= 7:
        historical = df['Global_active_power'].iloc[-7:].values
        hist_df = pd.DataFrame({
            "Day": [f"D-{7-i}" for i in range(7)],
            "kW": historical.round(2)
        })
        st.line_chart(hist_df.set_index("Day"), use_container_width=True)
    

    st.subheader(" Week-over-Week Comparison")
    if len(df) >= 7:
        last_week_avg = df['Global_active_power'].iloc[-7:].mean()
        next_week_avg = total_prediction / 7
        
        comparison_df = pd.DataFrame({
            "Period": ["Last Week", "Next Week"],
            "Average Daily (kW)": [last_week_avg, next_week_avg]
        })
        st.bar_chart(comparison_df.set_index("Period"), use_container_width=True)
        
        change = ((next_week_avg - last_week_avg) / last_week_avg) * 100
        if change > 0:
            st.info(f" Expected increase of {change:.1f}% compared to last week")
        else:
            st.info(f" Expected decrease of {abs(change):.1f}% compared to last week")
    
  
    st.subheader(" Prediction Distribution")
    dist_df = pd.DataFrame({
        "Value": preds.round(2)
    })
    st.area_chart(dist_df, use_container_width=True)
    

    st.markdown("---")
    st.subheader(" Download Results")
    
    result_df = pd.DataFrame({
        "Day": range(1, len(preds) + 1),
        "Date": pd.date_range(start=pd.Timestamp.now(), periods=len(preds), freq='D').strftime('%Y-%m-%d'),
        "Predicted (kW)": preds.round(2)
    })
    
    csv_result = result_df.to_csv(index=False)
    st.download_button(
        label=" Download Forecast as CSV",
        data=csv_result,
        file_name="electricity_forecast.csv",
        mime="text/csv"
    )

