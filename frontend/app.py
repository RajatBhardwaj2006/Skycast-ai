"""SkyCast Streamlit client; every prediction field maps to the saved model."""
from datetime import date, time
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("SKYCAST_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 20
st.set_page_config("SkyCast | AI Airfare Intelligence", "✈️", layout="wide")
st.markdown("""<style>
.stApp{background:#f3f8fa;color:#12303c}.hero{padding:1.4rem 0 .6rem}.hero h1{color:#073b4c;font-size:3rem;margin:.1rem 0}.eyebrow{color:#087e8b;font-size:.8rem;font-weight:700;letter-spacing:.12em}.panel{background:#fff;border:1px solid #d8e6ea;border-radius:16px;padding:1.2rem}.fare{font-size:2.8rem;font-weight:750;color:#073b4c;margin:.2rem 0}.stButton>button{background:#087e8b;color:white;border:0;border-radius:10px;font-weight:700;min-height:2.8rem}div[data-testid="stMetric"]{background:#fff;border:1px solid #d8e6ea;border-radius:12px;padding:.7rem}</style>""", unsafe_allow_html=True)


def get(path):
    response = requests.get(API_URL + path, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def dashboard_data():
    return get("/catalog"), get("/metrics"), get("/feature-importance"), get("/geo-experiment")


def band(value):
    h = value.hour
    return "Late Night" if h < 4 else "Early Morning" if h < 7 else "Morning" if h < 12 else "Afternoon" if h < 16 else "Evening" if h < 20 else "Night"


def error_message(response):
    try:
        detail = response.json().get("detail", "Prediction failed.")
        return detail.get("message", detail) if isinstance(detail, dict) else str(detail)
    except ValueError:
        return "Prediction failed. Please check the API and try again."


st.markdown("""<div class="hero"><div class="eyebrow">AI AIRFARE INTELLIGENCE PLATFORM</div><h1>SkyCast</h1><p>Historical airfare estimates from the trained Random Forest model—not live airline quotes.</p></div>""", unsafe_allow_html=True)
try:
    catalog, metrics, importance, geo = dashboard_data()
    ready = True
except requests.RequestException:
    ready, catalog, metrics, importance, geo = False, {}, {}, {}, {}
    st.error("SkyCast API is unavailable. Please start the backend.")

predict_tab, insights_tab, about_tab = st.tabs(["Predict fare", "Model insights", "Methodology"])
with predict_tab:
    if not ready:
        st.info("Run `uvicorn backend.app.main:app --reload` from the project root, then refresh.")
    else:
        cities = catalog["training_cities"]
        left, right = st.columns([1.08, .92], gap="large")
        with left:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.subheader("Flight details")
            st.caption("Travel date calculates the model's booking-window feature. Routes use cities present in training data.")
            with st.form("prediction"):
                c1, c2 = st.columns(2)
                origin = c1.selectbox("Origin", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
                destinations = [x for x in cities if x != origin]
                destination = c2.selectbox("Destination", destinations, index=destinations.index("Mumbai") if "Mumbai" in destinations else 0)
                c1, c2 = st.columns(2)
                airline = c1.selectbox("Airline", catalog["airlines"])
                travel_class = c2.selectbox("Travel class", catalog["classes"])
                c1, c2, c3 = st.columns(3)
                travel_date = c1.date_input("Travel date", value=date.today())
                departure = c2.time_input("Departure time", value=time(9, 0))
                arrival = c3.time_input("Arrival time", value=time(11, 15))
                c1, c2 = st.columns(2)
                duration = c1.number_input("Duration (hours)", .25, 48.0, 2.25, .25)
                stop_label = c2.selectbox("Stops", ["Non-stop", "1 stop", "2+ stops"])
                days_left = max(0, (travel_date - date.today()).days)
                st.caption(f"Booking window: **{days_left} days left**")
                submitted = st.form_submit_button("Predict fare", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            if not submitted:
                st.markdown("<div class='panel'><h3>Your estimate will appear here</h3><p>Choose flight details and run the trained model.</p></div>", unsafe_allow_html=True)
            else:
                payload = {"source_city": origin, "destination_city": destination, "airline": airline, "class": travel_class, "departure_time": band(departure), "arrival_time": band(arrival), "stops": {"Non-stop":"zero", "1 stop":"one", "2+ stops":"two_or_more"}[stop_label], "duration": duration, "days_left": days_left}
                with st.spinner("Running the trained model…"):
                    try:
                        response = requests.post(API_URL + "/predict", json=payload, timeout=TIMEOUT)
                        result = response.json() if response.ok else None
                        if not result: st.error(error_message(response))
                    except requests.RequestException:
                        result = None; st.error("SkyCast API is unavailable. Please start the backend.")
                if result:
                    price_range = result["expected_price_range"]
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.caption("ESTIMATED FARE")
                    st.markdown(f"<p class='fare'>₹{result['predicted_price']:,.0f}</p>", unsafe_allow_html=True)
                    st.caption(f"Expected model-error band: ₹{price_range['low']:,.0f} – ₹{price_range['high']:,.0f}")
                    st.info(result["confidence_note"])
                    a, b = st.columns(2)
                    a.metric("Route distance", f"{result['distance_km']:,.0f} km")
                    b.metric("Booking window", f"{result['summary']['days_left']} days")
                    st.write(f"**{result['source']['city']} → {result['destination']['city']}** · {result['summary']['airline']} · {result['summary']['class']} · {stop_label}")
                    st.caption(f"Model: {result['model']}. {price_range['note']}")
                    st.markdown("</div>", unsafe_allow_html=True)

with insights_tab:
    if ready:
        a, b, c = st.columns(3)
        a.metric("Held-out MAE", f"₹{metrics['mae']:,.0f}")
        b.metric("Held-out RMSE", f"₹{metrics['rmse']:,.0f}")
        c.metric("Held-out R²", f"{metrics['r2']:.4f}")
        left, right = st.columns(2)
        with left:
            st.subheader("Feature importance")
            frame = pd.DataFrame(importance.get("features", [])[:10])
            if not frame.empty: st.bar_chart(frame.set_index("feature")["importance"], horizontal=True)
            st.caption("Grouped impurity importance from the saved model. It is not a per-flight causal explanation.")
        with right:
            st.subheader("Geographic feature experiment")
            rows = geo.get("results", [])
            if rows: st.dataframe(pd.DataFrame(rows)[["model", "mae", "rmse", "r2"]], hide_index=True, use_container_width=True)
            st.caption(geo.get("note", ""))
        st.subheader("Model comparison")
        comparison = metrics.get("comparison", [])
        if comparison: st.dataframe(pd.DataFrame(comparison)[["model", "mae", "rmse", "r2"]], hide_index=True, use_container_width=True)

with about_tab:
    st.subheader("How it works")
    st.write("The pipeline uses airline, time bands, stops, class, cities, duration, booking window, route coordinates and haversine distance. It loads the trained model once in the API process.")
    st.subheader("Limitations")
    st.write("SkyCast does not query airline inventory or live prices. Training covers six Indian city markets; the price range is ± held-out MAE, not a confidence interval or guarantee.")
