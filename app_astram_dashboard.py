import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from catboost import CatBoostClassifier
import altair as alt
import os

# Set page structure
st.set_page_config(
    page_title="ASTraM Dispatch Center",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 ASTraM: AI-Driven Incident Dispatch Dashboard")
st.markdown("Automated Severity Classification and Resource Deployment for Bengaluru Traffic Police.")

# Load the OPTIMIZED 3-Tier model
@st.cache_resource
def load_trained_model():
    model_path = "astram_incident_classifier.cbm"
    if os.path.exists(model_path):
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model, True
    else:
        return None, False

model, model_loaded = load_trained_model()

if not model_loaded:
    st.error("⚠️ Model not found! Ensure 'astram_incident_classifier.cbm' is in the same folder.")

# ==========================================
# SIDEBAR CONTROLS (WITH MULTI-SELECT & "NONE")
# ==========================================
st.sidebar.header("🚨 Active Incident Feed")

event_cause_list = st.sidebar.multiselect(
    "Event Cause(s)", 
    ["None", "vehicle_breakdown", "pot_holes", "water_logging", "vip_movement", "rallies_and_events"],
    default=["vehicle_breakdown"]
)

veh_type_list = st.sidebar.multiselect(
    "Vehicle Type(s) Involved", 
    ["None", "bmtc_bus", "lcv", "truck", "two_wheeler", "car"],
    default=["bmtc_bus"]
)

event_type = st.sidebar.selectbox("Event Category", ["None", "unplanned", "planned"])
priority = st.sidebar.selectbox("Assigned Priority", ["None", "high", "medium", "low"])
corridor = st.sidebar.selectbox("Target Corridor Segment", ["None", "tumkur road", "outer ring road", "hosur road", "non-corridor"])
police_station = st.sidebar.selectbox("Responsible Police Station", ["jayanagara", "ashok nagar", "hebbala", "silk board", "madiwala"])
zone = st.sidebar.selectbox("Traffic Control Zone", ["None", "south", "east", "north", "west"])
incident_hour = st.sidebar.slider("Time of Incident (Hour)", 0, 23, 17)

# Approximate coordinates for the map
coord_mappings = {
    "jayanagara": [12.9299, 77.5800],
    "ashok nagar": [12.9720, 77.6194],
    "hebbala": [13.0354, 77.5978],
    "silk board": [12.9176, 77.6244],
    "madiwala": [12.9226, 77.6174]
}
selected_coords = coord_mappings.get(police_station, [12.9716, 77.5946])

# ==========================================
# PREDICTION ENGINE EXECUTION
# ==========================================
if 'engine_running' not in st.session_state:
    st.session_state.engine_running = False

if st.sidebar.button("🚨 Generate Deployment Plan"):
    st.session_state.engine_running = True

if st.session_state.engine_running and model_loaded:
    
    # --- SAFE EXTRACTION LOGIC FOR THE AI ---
    safe_cause = "unspecified" if not event_cause_list or "None" in event_cause_list else event_cause_list[0]
    safe_veh = "unspecified" if not veh_type_list or "None" in veh_type_list else veh_type_list[0]
    safe_type = "unspecified" if event_type == "None" else event_type
    safe_priority = "unspecified" if priority == "None" else priority
    safe_corridor = "unspecified" if corridor == "None" else corridor
    safe_zone = "unspecified" if zone == "None" else zone
    
    # 1. Feature Assembly
    time_frac = incident_hour + 0.5 
    is_peak = 1 if (8 <= incident_hour <= 11) or (17 <= incident_hour <= 20) else 0
    cause_priority = f"{safe_cause}_{safe_priority}"
    station_peak = f"{police_station}_{is_peak}"
    
    input_data = pd.DataFrame([{
        'latitude': selected_coords[0],
        'longitude': selected_coords[1],
        'hour': incident_hour, 
        'day_of_week': 2, 
        'month': 6,
        'is_weekend': 0,
        'hour_sin': np.sin(2 * np.pi * time_frac / 24.0),
        'hour_cos': np.cos(2 * np.pi * time_frac / 24.0),
        'is_peak_hour': is_peak,
        'event_type': safe_type,
        'event_cause': safe_cause,
        'priority': safe_priority,
        'veh_type': safe_veh,
        'corridor': safe_corridor,
        'police_station': police_station,
        'zone': safe_zone,
        'cause_priority': cause_priority,
        'station_peak': station_peak
    }])
    
    # 2. Get AI 3-Tier Prediction
    predicted_tier = model.predict(input_data)[0][0]
    
    # 3. Dynamic Deployment Logic based on the new 3 Tiers
    if "Tier 1" in predicted_tier:
        color = "green"
        radius = 500
        officers = 1
        tow = "No"
        alert = "✅ MINIMAL IMPACT: Local warden dispatch authorized. Standard clearing procedure."
    elif "Tier 2" in predicted_tier:
        color = "orange"
        radius = 1200
        officers = 2
        tow = "Standby"
        alert = "⚠️ MODERATE IMPACT: Manually clear adjacent intersections. Monitor spillover."
    else: # Tier 3 (Severe)
        color = "red"
        radius = 2500
        officers = 5
        tow = "Immediate Heavy Dispatch"
        alert = "🚨 SEVERE IMPACT ALERT: 90+ Minute gridlock expected. Barricade entry points and activate city-wide diversion routes."

    tier_label = predicted_tier.split(":")[0]
    duration_label = predicted_tier.split(":")[1].strip()

    # --- NEW FEATURE: HISTORICAL CONTEXT CHART ---
    st.subheader("📈 Real-Time Corridor Analytics")
    hours = np.arange(0, 24, 1)
    # Simulated bell curve for Bengaluru traffic volume
    traffic_vol = 100 * (np.exp(-0.5 * ((hours - 9) / 2)**2) + np.exp(-0.5 * ((hours - 18) / 2)**2)) + 20
    chart_data = pd.DataFrame({"Hour": hours, "Traffic Volume": traffic_vol})
    
    base = alt.Chart(chart_data).mark_line(color="gray").encode(x='Hour:Q', y='Traffic Volume:Q')
    point = alt.Chart(pd.DataFrame({'Hour': [incident_hour], 'Traffic Volume': [traffic_vol[incident_hour]]})).mark_circle(size=150, color=color).encode(x='Hour:Q', y='Traffic Volume:Q')
    
    st.altair_chart(base + point, use_container_width=True)

    # 4. Metrics Panel
    st.subheader("💡 AI Impact Assessment & Dispatch Plan")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(f"Severity ({tier_label})", duration_label)
    col2.metric("Required Personnel", f"{officers} Officers")
    col3.metric("Tow Assets", tow)
    col4.metric("Impact Radius", f"{radius} meters")
    
    if is_peak == 1:
        st.warning("⏱️ RUSH HOUR MODIFIER ACTIVE: The AI has factored peak hour traffic volumes into this prediction.")
    
    display_cause = ", ".join(event_cause_list) if "None" not in event_cause_list else "Unspecified Event"
    display_veh = ", ".join(veh_type_list) if "None" not in veh_type_list else "Unknown Vehicle"
    st.info(f"**Scenario:** {display_cause} involving {display_veh}. \n\n" + alert)

    # 5. Interactive Map Layout
    st.subheader("🗺️ Live Deployment Target Coordinate Map")
    
    m = folium.Map(location=selected_coords, zoom_start=13, tiles="CartoDB dark_matter")
    
    folium.Marker(
        selected_coords,
        popup=f"{tier_label} - {safe_cause}",
        icon=folium.Icon(color=color, icon="info-sign")
    ).add_to(m)
    
    folium.Circle(
        location=selected_coords,
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.4
    ).add_to(m)
    
    # Render map securely without refreshing
    st_folium(m, width=1100, height=500, returned_objects=[])

    # --- NEW FEATURE: AUTOMATED DISPATCH DRAFT ---
    st.markdown("---")
    st.subheader("📱 Automated Dispatch Communications")
    with st.expander("View Auto-Generated Radio / SMS Dispatch Draft", expanded=True):
        dispatch_text = f"""🚨 **ASTraM AUTOMATED DISPATCH ALERT** 🚨
**Location:** {police_station.upper()} JURISDICTION | {selected_coords[0]}, {selected_coords[1]}
**Incident:** {safe_cause.replace('_', ' ').title()} involving {safe_veh.replace('_', ' ').title()}
**Predicted Severity:** {tier_label} ({duration_label})

**ACTION REQUIRED:**
- Deploy {officers} Traffic Personnel immediately.
- Tow Support: {tow}.
- Establish {radius}m perimeter. 
- {alert.split(':')[1].strip() if ':' in alert else alert}"""
        
        st.code(dispatch_text, language="markdown")
        st.caption("Click the copy button in the top right of the code box to instantly paste this to the control room group.")
    
elif not st.session_state.engine_running:
    st.info("👈 Configure the incident in the sidebar and hit 'Generate Deployment Plan' to view the AI analysis.")