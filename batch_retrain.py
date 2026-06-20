import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import os
import warnings
warnings.filterwarnings('ignore')

print("🔄 Initiating ASTraM Nightly Micro-Batch Pipeline...")

historical_file = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
log_file = "astram_retraining_log.csv"

# 1. Load Historical Data
print("📊 Loading Historical Baseline...")
df_hist = pd.read_csv(historical_file)

# 2. Load New Feedback Data (If it exists)
if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
    print("📈 Found new real-world feedback logs! Merging with baseline...")
    df_new = pd.read_csv(log_file)
    # Align column names from the feedback log to the historical log
    df_new = df_new.rename(columns={
        'vehicle_type': 'veh_type',
        'actual_clearance_mins': 'impact_duration'
    })
    # Add safe default coordinates and enforce UTC Timezone
    df_new['latitude'] = 12.9716
    df_new['longitude'] = 77.5946
    df_new['start_datetime'] = pd.Timestamp.now(tz='UTC') # 🔥 FIX: Enforce UTC
    df_new['priority'] = 'unspecified'
    df_new['event_type'] = 'unspecified'
    df_new['corridor'] = 'unspecified'
    df_new['police_station'] = 'unspecified'
    df_new['zone'] = 'unspecified'
    df_new['authenticated'] = 'yes'
    
    # Merge the datasets
    df = pd.concat([df_hist, df_new], ignore_index=True)
else:
    print("⚠️ No new feedback logs found today. Retraining on historical baseline only.")
    df = df_hist

# ==========================================
# FEATURE ENGINEERING PIPELINE
# ==========================================
print("🛠️ Processing Geospatial and NLP Features...")

# 🔥 FIX: Enforce UTC on all dates so Pandas math doesn't crash
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)

if 'closed_datetime' in df.columns:
    df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce', utc=True)
    mask = df['impact_duration'].isna()
    df.loc[mask, 'impact_duration'] = (df.loc[mask, 'closed_datetime'] - df.loc[mask, 'start_datetime']).dt.total_seconds() / 60.0

df = df[(df['impact_duration'] >= 5) & (df['impact_duration'] <= 360)]

def assign_tier(minutes):
    if minutes <= 45: return "Tier 1: < 45 mins (Minor)"
    elif minutes <= 90: return "Tier 2: 45-90 mins (Moderate)"
    else: return "Tier 3: 90+ mins (Severe)"

df['severity_tier'] = df['impact_duration'].apply(assign_tier)

# Extract time features safely
df['hour'] = df['start_datetime'].dt.hour.fillna(17).astype(int)
df['day_of_week'] = df['start_datetime'].dt.dayofweek.fillna(2).astype(int)
df['month'] = df['start_datetime'].dt.month.fillna(6).astype(int)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

time_frac = df['hour'] + 0.5
df['hour_sin'] = np.sin(2 * np.pi * time_frac / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * time_frac / 24.0)

# Fill Categorical Missing Values
cat_features = ['event_type', 'event_cause', 'priority', 'veh_type', 'corridor', 'police_station', 'zone', 'requires_road_closure', 'authenticated']
for col in cat_features:
    if col not in df.columns:
        df[col] = 'unspecified'
    df[col] = df[col].astype(str).str.lower().str.strip().replace('nan', 'unspecified')

df['cause_priority'] = df['event_cause'] + "_" + df['priority']
df['station_peak'] = df['police_station'] + "_" + df['is_peak_hour'].astype(str)
cat_features.extend(['cause_priority', 'station_peak'])

# Setup NLP Feature
if 'description' not in df.columns:
    df['description'] = 'unspecified'
df['description'] = df['description'].astype(str).str.lower().fillna('unspecified')
text_features = ['description']

features = [
    'latitude', 'longitude', 'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'is_peak_hour', 'event_type', 'event_cause', 'priority', 
    'veh_type', 'corridor', 'police_station', 'zone', 'cause_priority', 'station_peak',
    'requires_road_closure', 'authenticated', 'description'
]

X = df[features]
y = df['severity_tier']

# ==========================================
# MODEL RETRAINING
# ==========================================
print(f"🐅 Retraining Core AI on {len(df)} total operational records...")

model = CatBoostClassifier(
    iterations=500, # Fast 30-second retrain for daily updates
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    text_features=text_features,
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    verbose=100
)

model.fit(X, y)

model.save_model("astram_incident_classifier.cbm")
print("\n✅ MLOps Pipeline Complete. The newly merged NLP model is live for the Command Center.")