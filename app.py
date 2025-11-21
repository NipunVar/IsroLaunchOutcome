# app.py (full updated)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import numpy as np
from matplotlib.ticker import FuncFormatter  # <-- added to format x-axis as percent

# --- 1. CONFIGURATION AND STYLING (SPACE TECH THEME) ---
st.set_page_config(
    page_title="ISRO Advanced Analytics Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKGROUND_COLOR = "#0b0c10"
CARD_BACKGROUND = "#1f2a40"
LIGHT_TEXT_COLOR = "#E0E7FF"
ACCENT_COLOR = "#4e8cfc"
SECONDARY_ACCENT = "#cb69c1"

st.markdown(f"""
<style>
.stApp {{
    background-color: {BACKGROUND_COLOR};
    color: {LIGHT_TEXT_COLOR};
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}
h1 {{
    color: {ACCENT_COLOR};
    text-align: center;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(78, 140, 252, 0.6), 0 0 10px rgba(78, 140, 252, 0.3);
}}
p, .stMarkdown, .stText, .stAlert, h2, h3, .stSidebar h2 {{
    color: {LIGHT_TEXT_COLOR};
}}
.stSidebar {{
    background-color: #161b22;
    border-right: 1px solid #3c4a63;
}}
.stSidebar h2 {{
    color: {SECONDARY_ACCENT};
    font-weight: 400;
}}
[data-testid="stMetric"] {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid #3c4a63;
    border-left: 5px solid {ACCENT_COLOR};
    border-radius: 6px;
    padding: 15px 0;
    text-align: center;
    box-shadow: 0 0 10px rgba(78, 140, 252, 0.1);
}}
[data-testid="stMetricLabel"] {{ font-size: 1.0em; color: #a0a8b8; }}
[data-testid="stMetricValue"] {{ color: {ACCENT_COLOR}; font-size: 2.2em; font-weight: 700; }}
.stPlotlyChart, .matplotlib {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid #3c4a63;
    border-radius: 6px;
    padding: 15px;
    box-shadow: 0 0 10px rgba(78, 140, 252, 0.1);
}}
/* force folium/leaflet map to occupy full width in Streamlit column */
.leaflet-container {{
    width: 100% !important;
    height: 800px !important;
}}
</style>
""", unsafe_allow_html=True)


# --- 2. DATA LOADING AND SYNTHETIC GENERATION (if needed) ---
@st.cache_data
def load_data(file_path):
    dummy_data_path = 'data/isro_300_missions.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(dummy_data_path):
        # generate synthetic dataset (keeps same columns as your pipeline expects)
        np.random.seed(42)
        n = 300
        vehicles = ["SLV", "ASLV", "PSLV", "GSLV", "GSLV Mk III", "SSLV"]
        vehicle_success_rate_map = {
            "SLV": 0.6, "ASLV": 0.64, "PSLV": 0.95, "GSLV": 0.85, "GSLV Mk III": 0.93, "SSLV": 0.75
        }
        orbit_types = ["LEO", "SSO", "GTO", "GEO", "MEO", "PO", "Sun-Synchronous", "Suborbital"]
        mission_types = ["Communication", "Earth Observation", "Navigation", "Scientific", "Technology Demo", "Interplanetary"]
        launch_windows = ["Morning", "Afternoon", "Evening", "Night"]

        launch_vehicle = np.random.choice(vehicles, size=n)
        orbit_type = np.random.choice(orbit_types, size=n)
        payload_weight_kg = np.random.uniform(50, 4500, n).round()
        temperature_C = np.random.normal(25, 10, n).clip(-10, 45).round(1)
        wind_speed_kmh = np.random.normal(10, 8, n).clip(0, 50).round(1)
        humidity_percent = np.random.uniform(10, 90, n).round(1)
        launch_window = np.random.choice(launch_windows, size=n)
        mission_type = np.random.choice(mission_types, size=n)
        system_health_index = np.random.beta(5, 2, n).round(3)
        vehicle_success_rate = [vehicle_success_rate_map[v] for v in launch_vehicle]

        launch_dates = pd.to_datetime(
            np.random.randint(
                pd.Timestamp('2000-01-01').value // 10**9,
                pd.Timestamp('2025-01-01').value // 10**9,
                n
            ), unit='s'
        )

        def simulate_outcome(system_health, success_rate):
            prob_success = system_health * success_rate
            return np.random.binomial(1, prob_success)

        launch_outcome = [simulate_outcome(sh, sr) for sh, sr in zip(system_health_index, vehicle_success_rate)]

        data = pd.DataFrame({
            "launch_date": launch_dates,
            "launch_vehicle": launch_vehicle,
            "orbit_type": orbit_type,
            "payload_weight_kg": payload_weight_kg,
            "temperature_C": temperature_C,
            "wind_speed_kmh": wind_speed_kmh,
            "humidity_percent": humidity_percent,
            "launch_window": launch_window,
            "mission_type": mission_type,
            "system_health_index": system_health_index,
            "vehicle_success_rate": vehicle_success_rate,
            "launch_outcome": launch_outcome
        })

        launch_sites = [
            'Satish Dhawan Space Centre', 'Vikram Sarabhai Space Centre', 'Thumba Launch Centre',
            'Chandipur Launch Site', 'Sriharikota Range'
        ]
        data['launch_site'] = np.random.choice(launch_sites, size=len(data))

        data.to_csv(dummy_data_path, index=False)

    try:
        df = pd.read_csv(file_path)
        if 'launch_date' in df.columns:
            df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading or generating data: {e}")
        st.stop()

data = load_data('data/isro_300_missions.csv')


# --- Feature engineering helper (must match training) ---
def feature_engineer(df):
    df = df.copy()
    # ensure launch_date datetime
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
        df['launch_month'] = df['launch_date'].dt.month.fillna(1).astype(int)
        df['launch_quarter'] = df['launch_date'].dt.quarter.fillna(1).astype(int)
    else:
        df['launch_month'] = df.get('launch_month', 1)
        df['launch_quarter'] = df.get('launch_quarter', 1)

    # default orbit_type if missing
    if 'orbit_type' not in df.columns:
        df['orbit_type'] = 'LEO'

    orbit_complexity_map = {
        'Suborbital': 1, 'LEO': 2, 'SSO': 3, 'PO': 3, 'MEO': 4, 'GTO': 5, 'GEO': 6, 'Interplanetary': 7
    }
    mission_type_complexity_map = {
        'Technology Demo': 1, 'Navigation': 2, 'Earth Observation': 3, 'Communication': 4, 'Scientific': 5, 'Interplanetary': 6, 'Manned': 6
    }

    df['orbit_score'] = df['orbit_type'].map(orbit_complexity_map).fillna(0).astype(float)
    df['mission_score'] = df['mission_type'].map(mission_type_complexity_map).fillna(0).astype(float)
    df['mission_complexity_score'] = df['orbit_score'] + df['mission_score']

    launch_window_risk_map = {'Morning': 1.0, 'Afternoon': 1.1, 'Evening': 1.5, 'Night': 1.3, 'Day': 1.0}
    df['launch_window_risk_index'] = df.get('launch_window', pd.Series()).map(launch_window_risk_map).fillna(1.0).astype(float)

    df.drop(columns=['orbit_score', 'mission_score'], inplace=True, errors='ignore')
    return df

# Precompute features on the dataset (used for defaults, alternatives, and dummy fitting)
data = feature_engineer(data)


# --- Model loading / creation (no banner printed to UI) ---
@st.cache_resource
def load_model(file_paths=None):
    if file_paths is None:
        file_paths = ['isro_launch_model_v2.pkl', 'launch_model_pipeline.pkl', 'isro_launch_model.pkl', 'launch_model.pkl']

    for fp in file_paths:
        if os.path.exists(fp):
            try:
                model = joblib.load(fp)
                # Do not show a banner in UI per request — just print to console
                print(f"Loaded trained model from: {fp}")
                return model, True
            except Exception as e:
                print(f"Found model file {fp} but failed to load: {e}")

    # If not found, build and fit a demo pipeline so ColumnTransformer attributes exist
    print("Trained model not found. Building and fitting a demo pipeline (for UI/demo).")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import StackingClassifier
    from xgboost import XGBClassifier

    numerical_features = [
        'payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent',
        'system_health_index', 'vehicle_success_rate',
        'launch_month', 'launch_quarter', 'mission_complexity_score', 'launch_window_risk_index'
    ]
    categorical_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site', 'orbit_type']

    num_feats = [f for f in numerical_features if f in data.columns]
    cat_feats = [f for f in categorical_features if f in data.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
        ],
        remainder='drop'
    )

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ('svc', SVC(kernel='linear', probability=True, random_state=42)),
        ('lr', LogisticRegression(solver='liblinear', random_state=42)),
    ]

    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        cv=3,
        n_jobs=-1
    )

    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', stacking_classifier)])

    # Fit dummy pipeline on sample data
    try:
        X_train = data[num_feats + cat_feats].copy()
        for c in cat_feats:
            X_train[c] = X_train[c].astype(str)
        y_train = data['launch_outcome']
        pipeline.fit(X_train, y_train)
        print("Demo pipeline fitted on sample data.")
        return pipeline, False
    except Exception as e:
        print(f"Failed to fit demo pipeline: {e}")
        return None, False

model, model_is_trained = load_model()


# prepare input for model (applies same feature engineering)
def prepare_input_for_model(df):
    df_proc = df.copy()
    if 'launch_date' not in df_proc.columns:
        df_proc['launch_date'] = pd.NaT
    df_proc = feature_engineer(df_proc)
    for c in ['launch_vehicle', 'mission_type', 'launch_window', 'launch_site', 'orbit_type']:
        if c in df_proc.columns:
            df_proc[c] = df_proc[c].astype(str)
    return df_proc


# --- 3. PLOTTING UTILITY ---
def apply_plot_style(fig, ax):
    ax.set_facecolor(CARD_BACKGROUND)
    fig.patch.set_facecolor(CARD_BACKGROUND)
    plt.rcParams['text.color'] = LIGHT_TEXT_COLOR
    plt.rcParams['axes.labelcolor'] = LIGHT_TEXT_COLOR
    plt.rcParams['xtick.color'] = LIGHT_TEXT_COLOR
    plt.rcParams['ytick.color'] = LIGHT_TEXT_COLOR
    plt.rcParams['axes.edgecolor'] = LIGHT_TEXT_COLOR
    try:
        plt.grid(axis='y', color='#3c4a63', linestyle='--', alpha=0.5)
    except:
        pass


# --- 4. SIDEBAR NAVIGATION ---
if 'selected_menu' not in st.session_state:
    st.session_state['selected_menu'] = "Project Overview"

st.sidebar.title("ISRO Advanced Analytics Platform")
st.sidebar.markdown("---")

ALL_MENU_ITEMS = [
    "Project Overview", "Real-Time Predictions", "Strategic Financials", "Benchmarking",
    "Feature Distributions", "Correlations", "Grouped Analysis",
    "Time Trends", "Raw Data Overview"
]

st.sidebar.markdown("### Core Modules")
selected_menu = st.sidebar.radio(
    "Select Analysis Module",
    options=ALL_MENU_ITEMS,
    index=ALL_MENU_ITEMS.index(st.session_state.selected_menu),
    key='main_navigation_radio'
)
st.session_state.selected_menu = selected_menu

st.title("ISRO Rocket Launch Analytics & Forecasting")
st.markdown("---")


# -----------------------------------------------------------------
# PROJECT OVERVIEW
# -----------------------------------------------------------------
if selected_menu == "Project Overview":
    st.header("Integrated Strategic Launch Analytics")
    st.subheader("Project Scope")
    st.markdown("""
This project consolidates decades of ISRO launch data into a single, interactive analytics platform designed to assist engineers, mission planners and strategic stakeholders. It combines careful data engineering, domain-driven feature construction and machine learning to estimate mission success probabilities and highlight the highest-impact risk factors. The dataset contains environmental telemetry (temperature, wind, humidity), vehicle characteristics (launch vehicle family, historical vehicle success rates), mission descriptors (mission type, orbit type) and system readiness metrics (system health index).  

Using this foundation we construct advanced features — for example, a mission complexity score that blends orbit and mission-type difficulty, and a launch-window risk index that captures operational timing constraints — ensuring the model understands real operational risks, not just raw sensor values. The modeling pipeline employs robust preprocessing (scaling, one-hot encoding) and a stacking ensemble of complementary classifiers to balance bias and variance while providing high-quality probability estimates.  

The dashboard offers both single-run, real-time predictions and batch processing for program-level scenario evaluation. Interactive sensitivity analysis reveals which inputs change the probability the most for a given mission — enabling engineers to see whether payload mass, thermal conditions, or subsystem health matter most for a particular launch. Supplementary modules include benchmarking against industry cost-per-kilogram, Monte Carlo scenario simulation for financial risk, and geospatial visualizations to support launch-site planning and debris-risk assessment.  

Overall, the platform is designed to be reproducible and auditable: preprocessing and feature engineering match the training logic, model artifacts are loadable in production, and visual explanations support transparent decision-making for high-stakes aerospace operations.
    """)
    colA, colB, colC = st.columns(3)
    colA.metric("Total Missions Analyzed", f"{len(data)}", "Data snapshot")
    colB.metric("Overall Success Rate", f"{data['launch_outcome'].mean():.2%}", "Historical KPI")
    top_vehicle = data['launch_vehicle'].mode()[0] if 'launch_vehicle' in data.columns else "N/A"
    colC.metric("Top Launch Vehicle", top_vehicle, "Most frequent in dataset")


# -----------------------------------------------------------------
# REAL-TIME PREDICTIONS
# -----------------------------------------------------------------
elif selected_menu == "Real-Time Predictions":
    st.header("Real-Time Mission Success Prediction")
    if model is None:
        st.warning("Prediction feature is unavailable because the model failed to load or construct.")
    else:
        with st.form("prediction_form", border=False):
            st.subheader("Mission Input Parameters")
            col1, col2 = st.columns(2)

            with col1:
                launch_vehicle = st.selectbox("Launch Vehicle", sorted(data['launch_vehicle'].unique()))
                mission_type = st.selectbox("Mission Type", sorted(data['mission_type'].unique()))
                launch_window = st.selectbox("Launch Window", sorted(data['launch_window'].unique()))
                launch_site = st.selectbox("Launch Site", sorted(data['launch_site'].unique()))
                orbit_type = st.selectbox("Orbit Type (optional)", sorted(data['orbit_type'].unique()) if 'orbit_type' in data.columns else ['LEO','GTO','SSO','GEO'], index=0)

            with col2:
                payload_weight_kg = st.number_input("Payload Weight (kg)", 50, 4500, 1500)
                temperature_C = st.slider("Temperature (°C)", float(data['temperature_C'].min()), float(data['temperature_C'].max()), 28.0)
                wind_speed_kmh = st.slider("Wind Speed (km/h)", float(data['wind_speed_kmh'].min()), float(data['wind_speed_kmh'].max()), 15.0)
                system_health_index = st.slider("System Health Index (0-1)", 0.0, 1.0, 0.95)

                vehicle_success_rate_map = data.set_index('launch_vehicle')['vehicle_success_rate'].to_dict()
                vehicle_success_rate = vehicle_success_rate_map.get(launch_vehicle, 0.90)
                humidity_percent = data['humidity_percent'].mean()

            submitted = st.form_submit_button("Initiate Prediction", type="primary")

        if submitted:
            input_df = pd.DataFrame({
                'launch_vehicle': [launch_vehicle],
                'mission_type': [mission_type],
                'launch_window': [launch_window],
                'launch_site': [launch_site],
                'orbit_type': [orbit_type],
                'payload_weight_kg': [payload_weight_kg],
                'temperature_C': [temperature_C],
                'wind_speed_kmh': [wind_speed_kmh],
                'humidity_percent': [humidity_percent],
                'system_health_index': [system_health_index],
                'vehicle_success_rate': [vehicle_success_rate],
                # optional: 'launch_date' not provided — feature_engineer will handle defaults
            })

            # Prepare (feature-engineer) input
            safe_input = prepare_input_for_model(input_df)

            # Convert categorical dtypes
            for c in ['launch_vehicle', 'mission_type', 'launch_window', 'launch_site', 'orbit_type']:
                if c in safe_input.columns:
                    safe_input[c] = safe_input[c].astype(str)

            # Predict
            try:
                prob = model.predict_proba(safe_input)[0][1]
                st.metric("Predicted Success Probability", f"{prob:.2%}", delta=None)
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
                try:
                    preproc = model.named_steps.get('preprocessor', None)
                    if preproc is not None and hasattr(preproc, 'get_feature_names_out'):
                        exp = list(preproc.get_feature_names_out())
                        st.info(f"Model expected features (example subset): {exp[:30]}{'...' if len(exp)>30 else ''}")
                except Exception:
                    pass

            # --- LOCAL SENSITIVITY (Interactive "Top Risk & Success Drivers") ---
            st.markdown("---")
            st.subheader("Top Risk & Success Drivers (local sensitivity)")

            def local_sensitivity_scores(model, base_df, data_ref, numeric_delta=0.05):
                baseline = base_df.copy()
                baseline_prepared = prepare_input_for_model(baseline)
                try:
                    base_prob = float(model.predict_proba(baseline_prepared)[0][1])
                except Exception:
                    return pd.DataFrame(columns=['Feature','Delta','AbsDelta'])

                scores = []
                row = baseline.iloc[0]

                for col in baseline.columns:
                    if col in ['launch_date', 'mission_id']:
                        continue

                    # numeric columns
                    if pd.api.types.is_numeric_dtype(baseline[col]) and col not in ['launch_month','launch_quarter']:
                        val = float(row[col]) if pd.notna(row[col]) else 0.0
                        if abs(val) < 1:
                            delta = numeric_delta
                            pert_up = val + delta
                            pert_dn = max(val - delta, 0.0)
                        else:
                            delta = max(abs(val) * numeric_delta, 1e-3)
                            pert_up = val + delta
                            pert_dn = val - delta

                        up_df = baseline.copy(); dn_df = baseline.copy()
                        up_df[col] = pert_up; dn_df[col] = pert_dn
                        try:
                            up_prob = float(model.predict_proba(prepare_input_for_model(up_df))[0][1])
                            dn_prob = float(model.predict_proba(prepare_input_for_model(dn_df))[0][1])
                            signed_change = 0.5 * ((up_prob - base_prob) - (base_prob - dn_prob))
                            scores.append((col, signed_change, abs(signed_change)))
                        except Exception:
                            continue

                    else:
                        # categorical: swap to an alternative from data_ref
                        try:
                            alternatives = list(data_ref[col].dropna().astype(str).unique())
                        except Exception:
                            alternatives = []
                        current = str(row[col]) if pd.notna(row[col]) else None
                        alt = None
                        for a in alternatives:
                            if a != current:
                                alt = a; break
                        if alt is None:
                            if 'orbit' in col.lower():
                                alt = 'GTO' if current != 'GTO' else 'LEO'
                            elif 'window' in col.lower():
                                alt = 'Night' if current != 'Night' else 'Morning'
                            else:
                                continue
                        alt_df = baseline.copy()
                        alt_df[col] = alt
                        try:
                            alt_prob = float(model.predict_proba(prepare_input_for_model(alt_df))[0][1])
                            signed_change = alt_prob - base_prob
                            scores.append((col, signed_change, abs(signed_change)))
                        except Exception:
                            continue

                if len(scores) == 0:
                    return pd.DataFrame(columns=['Feature','Delta','AbsDelta'])
                df_scores = pd.DataFrame(scores, columns=['Feature','Delta','AbsDelta']).sort_values('AbsDelta', ascending=False).reset_index(drop=True)
                return df_scores

            try:
                sensitivity_df = local_sensitivity_scores(model, safe_input, data, numeric_delta=0.05)
                if sensitivity_df.empty:
                    st.info("Feature sensitivity unavailable for the current model/input.")
                else:
                    top_n = sensitivity_df.head(10).copy()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = top_n['Delta'].apply(lambda x: ACCENT_COLOR if x >= 0 else SECONDARY_ACCENT)
                    ax.barh(top_n['Feature'], top_n['Delta'], color=colors)
                    ax.set_xlabel("Change in Predicted Success Probability (signed)")
                    ax.set_title("Local Sensitivity: How changing each feature affects predicted success", color=LIGHT_TEXT_COLOR)

                    # Use FuncFormatter to format x-axis as percentages (avoids set_xticklabels warnings)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))

                    apply_plot_style(fig, ax)
                    st.pyplot(fig)
                    # Show numeric readout
                    display_df = top_n.assign(**{'Delta (%)': (top_n['Delta'] * 100).round(2)}).drop(columns=['AbsDelta'])
                    st.dataframe(display_df)
            except Exception as e:
                st.error(f"Could not compute sensitivity plot: {e}")

            # --- Batch Prediction Utility ---
            st.markdown("---")
            st.subheader("Batch Prediction Utility")
            uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    batch_safe = prepare_input_for_model(batch_df)
                    for c in ['launch_vehicle', 'mission_type', 'launch_window', 'launch_site', 'orbit_type']:
                        if c in batch_safe.columns:
                            batch_safe[c] = batch_safe[c].astype(str)
                    preds = model.predict_proba(batch_safe)[:, 1]
                    batch_safe['predicted_success_probability'] = preds
                    st.dataframe(batch_safe)
                    st.download_button("Download Predictions CSV", data=batch_safe.to_csv(index=False).encode(), file_name='batch_predictions.csv')
                except Exception as e:
                    st.error(f"Batch prediction failed. Error: {e}")


# -----------------------------------------------------------------
# STRATEGIC FINANCIALS
# -----------------------------------------------------------------
elif selected_menu == "Strategic Financials":
    st.header("Financial Strategy and Risk Assessment")
    try:
        with open('reusability_metrics.json', 'r') as f:
            reusability_metrics = json.load(f)
    except Exception:
        reusability_metrics = None
    try:
        with open('mc_results.json', 'r') as f:
            mc_results = json.load(f)
    except Exception:
        mc_results = None

    st.subheader("1. Reusable Launch Vehicle (RLV) ROI")
    if reusability_metrics:
        colA, colB, colC = st.columns(3)
        colA.metric("Total Disposable Cost (15 Missions)", f"${reusability_metrics['total_disposable_cost_M']}M")
        colB.metric("Total RLV Program Cost (15 Missions)", f"${reusability_metrics['total_reusable_cost_M']}M", delta=f"Savings: ${reusability_metrics['total_savings_M']}M", delta_color="inverse")
        colC.metric("Financial Break-Even Point", f"{reusability_metrics['break_even_point_missions']} Launches")
        st.info(reusability_metrics.get('roi_strategic_insight', ''))
    else:
        st.error("Reusability metrics not found. Run reusability analysis scripts to generate them.")

    st.markdown("---")
    st.subheader("2. Monte Carlo Risk Assessment (High-Stakes Mission)")
    if mc_results:
        colD, colE, colF = st.columns(3)
        colD.metric("Simulated Success Rate", f"{mc_results['simulated_success_rate']:.2%}")
        colE.metric("Expected Net Value (ROI)", f"${mc_results['expected_net_value_M']}M")
        colF.metric("Value-at-Risk (95% VaR)", f"${mc_results['value_at_risk_M']}M")
        st.warning(mc_results.get('risk_insight', ''))
    else:
        st.error("Monte Carlo results not found. Run scenario simulator to generate them.")


# -----------------------------------------------------------------
# BENCHMARKING
# -----------------------------------------------------------------
elif selected_menu == "Benchmarking":
    st.header("Industry Benchmarking and Competitiveness")
    benchmark_chart = 'benchmarking_chart.png'
    if os.path.exists(benchmark_chart):
        st.image(benchmark_chart, caption="ISRO's Cost/Kg and Success Rate vs. Global Agencies")
    else:
        st.subheader("Cost/kg to LEO (Simulated Benchmarks)")
        benchmark_data = {
            'Agency': ['ISRO', 'SpaceX (F9)', 'NASA (SLS)', 'Roscosmos (Soyuz)'],
            'Cost per kg (USD)': [3000, 2700, 15000, 4500]
        }
        df_bench = pd.DataFrame(benchmark_data)
        st.line_chart(df_bench.set_index('Agency'))
        st.info(f"Placeholder chart used; run benchmarking_analysis.py to create '{benchmark_chart}'.")




# -----------------------------------------------------------------
# FEATURE DISTRIBUTIONS
# -----------------------------------------------------------------
elif selected_menu == "Feature Distributions":
    st.header("Key Feature Distributions")
    num_cols = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh']
    cat_cols = ['launch_vehicle', 'mission_type', 'launch_site']
    st.subheader("Numerical Feature Histograms")
    cols = st.columns(len(num_cols))
    for i, col in enumerate(num_cols):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(data[col], kde=True, ax=ax, color=ACCENT_COLOR, edgecolor='#000')
            ax.set_title(col, color=LIGHT_TEXT_COLOR)
            apply_plot_style(fig, ax)
            st.pyplot(fig)
    st.subheader("Categorical Feature Counts")
    cols = st.columns(len(cat_cols))
    for i, col in enumerate(cat_cols):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(y=data[col], ax=ax, palette='icefire', hue=data[col], legend=False)
            ax.set_title(col, color=LIGHT_TEXT_COLOR)
            apply_plot_style(fig, ax)
            st.pyplot(fig)


# -----------------------------------------------------------------
# CORRELATIONS
# -----------------------------------------------------------------
elif selected_menu == "Correlations":
    st.header("Numerical Feature Correlation Heatmap")
    num_data = data[['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate', 'launch_outcome']]
    corr_matrix = num_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, linecolor=BACKGROUND_COLOR, ax=ax, annot_kws={'color': LIGHT_TEXT_COLOR})
    ax.set_title("Full Feature Correlation Matrix", color=LIGHT_TEXT_COLOR)
    apply_plot_style(fig, ax)
    st.pyplot(fig)


# -----------------------------------------------------------------
# GROUPED ANALYSIS
# -----------------------------------------------------------------
elif selected_menu == "Grouped Analysis":
    st.header("Mission Success Rate by Launch Vehicle")
    success_rate = data.groupby('launch_vehicle')['launch_outcome'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=success_rate.index, y=success_rate.values, palette='magma', ax=ax)
    ax.set_title("Average Launch Success Rate by Vehicle Type", color=LIGHT_TEXT_COLOR)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    apply_plot_style(fig, ax)
    st.pyplot(fig)
    st.info(f"**Highest Success Rate:** {success_rate.index[0]} at {success_rate.iloc[0]:.2%}.")


# -----------------------------------------------------------------
# TIME TRENDS
# -----------------------------------------------------------------
elif selected_menu == "Time Trends":
    st.header("Rolling Average of Mission Success Over Time")
    time_data = data.sort_values('launch_date').copy()
    time_data['Year'] = time_data['launch_date'].dt.year
    time_data['Rolling_Success'] = time_data['launch_outcome'].rolling(window=10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='launch_date', y='Rolling_Success', data=time_data, ax=ax, color=ACCENT_COLOR, linewidth=3)
    ax.set_title("10-Mission Rolling Success Rate (Historical Trend)", color=LIGHT_TEXT_COLOR)
    ax.set_ylim(0.5, 1.05)
    mean_success = data['launch_outcome'].mean()
    ax.axhline(mean_success, color=SECONDARY_ACCENT, linestyle='--', label=f'Overall Mean ({mean_success:.2%})')
    ax.legend()
    apply_plot_style(fig, ax)
    st.pyplot(fig)


# -----------------------------------------------------------------
# RAW DATA OVERVIEW
# -----------------------------------------------------------------
elif selected_menu == "Raw Data Overview":
    st.header("Raw Mission Data Overview and Statistics")
    st.dataframe(data.head(20))
    st.subheader("Data Distribution Summary (Numerical Features)")
    st.dataframe(data.describe().T)
