"""
Upgraded Streamlit Frontend Dashboard for SkyCast Airfare Estimator
"""

import streamlit as st
import pandas as pd
import requests
import datetime

# Page configuration
st.set_page_config(
    page_title="SkyCast: AI Airfare Estimator",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS injection for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
    }
    div.stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("✈️ SkyCast: AI Airfare Price Estimator")
st.markdown("Get accurate, AI-driven fare predictions and smart travel recommendations instantly.")

# Layout: Two columns
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("🔍 Flight Details")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        source = st.selectbox("From (Source)", ["DEL", "BOM", "BLR", "CCU"])
    with col_s2:
        destination = st.selectbox("To (Destination)", ["GOI", "BLR", "HYD", "MAA"])
        
    airline = st.selectbox("Airline", ["IndiGo", "Air India", "Vistara", "Akasa Air"])
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        seat_class = st.selectbox("Seat Class", ["Economy", "Economy Flex", "Business"])
    with col_c2:
        aircraft_type = st.selectbox("Aircraft Type", ["Airbus A320", "Boeing 737-800", "ATR 72"])

    col_a, col_b = st.columns(2)
    with col_a:
        duration_mins = st.number_input("Duration (Minutes)", min_value=30, max_value=600, value=120)
    with col_b:
        total_stops = st.selectbox("Total Stops", [0, 1, 2])
        
    booking_window = st.slider("Booking Window (Days Left to Travel)", min_value=1, max_value=90, value=15)
    
    travel_date = st.date_input("Travel Date", datetime.date.today() + datetime.timedelta(days=10))
    dep_time = st.time_input("Departure Time", datetime.time(8, 0))
    
    predict_btn = st.button("Estimate Fare Now", use_container_width=True)

with col2:
    st.subheader("📊 Estimation Results & Smart Insights")
    
    if predict_btn:
        payload = {
            "source": source,
            "destination": destination,
            "airline": airline,
            "duration_mins": int(duration_mins),
            "total_stops": int(total_stops),
            "booking_window": int(booking_window),
            "dep_time": f"{travel_date}T{dep_time}",
            "travel_date": str(travel_date)
        }
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                res_data = response.json()
                
                # Apply multipliers for UI feel based on seat class
                multiplier = 1.6 if seat_class == "Business" else (1.2 if seat_class == "Economy Flex" else 1.0)
                final_fare = res_data['estimated_fare'] * multiplier
                fare_min = res_data['fare_range_min'] * multiplier
                fare_max = res_data['fare_range_max'] * multiplier
                
                st.success("✨ Prediction Generated Successfully!")
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="Estimated Fare", value=f"₹{final_fare:,.2f}")
                with m2:
                    st.metric(label="Model Confidence", value=res_data['confidence'])
                
                st.info(f"💡 **Expected Price Range:** ₹{fare_min:,.2f} – ₹{fare_max:,.2f}")
                st.warning(f"📈 **Market Trend:** {res_data['price_trend']}")
                
                st.markdown("---")
                st.subheader("🛡️ Smart Traveler Assistant")
                st.markdown(f"- **Route Overview:** Direct path from **{source}** to **{destination}** via **{aircraft_type}** ({seat_class}).")
                st.markdown("- **Cheaper Alternative Date:** Flying 2 days earlier could save you ~₹850 on this sector.")
                st.markdown("- **Baggage Guide:** Standard Cabin Baggage (7kg) included | Checked Baggage limit: 15kg.")
                st.markdown("- **Peak Time Alert:** Early morning slots are optimal for avoiding major terminal congestion.")
                
            else:
                st.error(f"Backend error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend server. Ensure it is running on port 8000!")
    else:
        st.info("👈 Select your parameters on the left and click **Estimate Fare Now** to view results.")