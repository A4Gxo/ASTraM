import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import optuna
import warnings
warnings.filterwarnings('ignore')

print("⏳ [1/5] Loading Raw ASTraM Dataset...")
df = pd.read_csv("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# ==========================================
# 1. 3-TIER TARGET & FEATURE ENGINEERING
# ==========================================
print("🛠️ [2/5] Engineering NLP & Geospatial Features...")

df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
df = df.dropna(subset=['start_datetime', 'closed_datetime'])

df['impact_duration'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
df = df[(df['impact_duration'] >= 5) & (df['impact_duration'] <= 360)]

def assign_tier_v2(minutes):
    if minutes <= 45: return "Tier 1: < 45 mins (Minor)"
    elif minutes <= 90: return "Tier 2: 45-90 mins (Moderate)"
    else: return "Tier 3: 90+ mins (Severe)"

df['severity_tier'] = df['impact_duration'].apply(assign_tier_v2)

# Temporal Features
df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek
df['month'] = df['start_datetime'].dt.month
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

time_frac = df['hour'] + (df['start_datetime'].dt.minute / 60.0)
df['hour_sin'] = np.sin(2 * np.pi * time_frac / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * time_frac / 24.0)

# Geospatial Boundaries
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])
df = df[(df['latitude'] > 12.0) & (df['latitude'] < 14.0)]
df = df[(df['longitude'] > 77.0) & (df['longitude'] < 79.0)]

# 🔥 NEW: Extract the High-Value Structural Columns
df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.lower()
df['authenticated'] = df['authenticated'].astype(str).str.lower()

cat_features = [
    'event_type', 'event_cause', 'priority', 'veh_type', 'corridor', 
    'police_station', 'zone', 'requires_road_closure', 'authenticated'
]

for col in cat_features:
    df[col] = df[col].astype(str).str.lower().str.strip().replace('nan', 'unspecified')

# Feature Crosses
df['station_peak'] = df['police_station'] + "_" + df['is_peak_hour'].astype(str)
df['cause_priority'] = df['event_cause'] + "_" + df['priority']
cat_features.extend(['cause_priority', 'station_peak'])

# 🔥 NEW: Natural Language Processing (NLP) Setup
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. OPTUNA HYPERPARAMETER SEARCH
# ==========================================
print("🤖 [3/5] Initiating Optuna AI Search (Now analyzing raw text!)...")

def objective(trial):
    param = {
        'iterations': 1000, 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'cat_features': cat_features,
        'text_features': text_features, # Telling CatBoost to read the text!
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'task_type': 'CPU',
        'verbose': False                              
    }
    
    model = CatBoostClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        use_best_model=True
    )
    return model.get_best_score()['validation']['Accuracy']

optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25) # Slightly bumped up trials to account for NLP

print(f"🏆 Optuna Search Complete! Best Validation Accuracy Found: {study.best_value * 100:.2f}%")

# ==========================================
# 3. FINAL TRAINING WITH BEST PARAMS
# ==========================================
print("🐅 [4/5] Training Final NLP Model with AI-Discovered Parameters...")

best_params = study.best_params
best_params['iterations'] = 2500 
best_params['cat_features'] = cat_features
best_params['text_features'] = text_features
best_params['loss_function'] = 'MultiClass'
best_params['eval_metric'] = 'Accuracy'
best_params['auto_class_weights'] = 'Balanced'
best_params['random_seed'] = 42
best_params['verbose'] = 500

final_model = CatBoostClassifier(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=200,
    use_best_model=True
)

# ==========================================
# 4. PERFORMANCE EVALUATION
# ==========================================
val_preds = final_model.predict(X_val)

print(f"\n💎 [5/5] Final Assessment complete!")
print(f"   -> Overall Strict Accuracy: {accuracy_score(y_val, val_preds) * 100:.2f}%")
print("\n📊 Detailed Tier Breakdown:")
print(classification_report(y_val, val_preds))

final_model.save_model("astram_incident_classifier.cbm")
print("💾 NLP Optuna Model exported to 'astram_incident_classifier.cbm' successfully!")

# ==========================================
# 5. THE HACKATHON PITCH METRIC (ADJACENT ACCURACY)
# ==========================================
print("\n🔥 --- EXECUTIVE PITCH METRICS --- 🔥")

y_val_numeric = y_val.str.extract(r'Tier (\d)')[0].astype(int)
val_preds_numeric = pd.Series(val_preds.flatten()).str.extract(r'Tier (\d)')[0].astype(int)

strict_correct = (y_val_numeric.values == val_preds_numeric.values).sum()
adjacent_correct = (abs(y_val_numeric.values - val_preds_numeric.values) <= 1).sum()
total_predictions = len(y_val_numeric)

strict_acc = (strict_correct / total_predictions) * 100
adjacent_acc = (adjacent_correct / total_predictions) * 100

print(f"Strict Machine Learning Accuracy: {strict_acc:.2f}%")
print(f"Operational 'Adjacent' Accuracy (Off by <= 1 Tier): {adjacent_acc:.2f}%")