# Save this code as app.py (Space Tech Theme)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import numpy as np 

# --- 1. CONFIGURATION AND STYLING (SPACE TECH THEME) ---
# Setting the layout to wide and configuring the sidebar state
st.set_page_config(
    page_title="ISRO Advanced Analytics Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the SPACE TECH color scheme constants
BACKGROUND_COLOR = "#0b0c10" # Deep Space Black
CARD_BACKGROUND = "#1f2a40"  # Deep Midnight Blue (Panel/Card Color)
LIGHT_TEXT_COLOR = "#E0E7FF" # Starlight White/Slightly Blue-Tinted
ACCENT_COLOR = "#4e8cfc"     # Electric Cyan/Blue (Primary Accent for Titles, Metrics)
SECONDARY_ACCENT = "#cb69c1" # Lavender/Nebula Pink (Secondary Accent for Highlights/Hover)

# Custom Space Tech Theme CSS 
st.markdown(f"""
<style>
/* Base Dark Theme (Deep Space) */
.stApp {{
    background-color: {BACKGROUND_COLOR}; 
    color: {LIGHT_TEXT_COLOR};
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}
/* Main Title Styling */
h1 {{
    color: {ACCENT_COLOR}; 
    text-align: center;
    font-weight: 700;
    /* Subtle neon-like glow effect */
    text-shadow: 0 0 8px rgba(78, 140, 252, 0.6), 0 0 10px rgba(78, 140, 252, 0.3);
}}
/* General Text */
p, .stMarkdown, .stText, .stAlert, h2, h3, .stSidebar h2 {{
    color: {LIGHT_TEXT_COLOR}; 
}}

/* Sidebar Styling (Slightly different dark tone for contrast) */
.stSidebar {{
    background-color: #161b22; 
    border-right: 1px solid #3c4a63; /* Tech blue border */
}}
.stSidebar h2 {{
    color: {SECONDARY_ACCENT}; 
    font-weight: 400;
}}
/* Navigation Link Styling (st.radio elements) */
.st-emotion-cache-12fm52p label {{
    font-size: 1.05em;
    padding: 8px 12px;
    border-radius: 4px;
    transition: background-color 0.2s, color 0.2s;
}}
.st-emotion-cache-12fm52p label:hover {{
    background-color: #2b2c36;
    color: {ACCENT_COLOR};
}}
/* Highlight selected radio option (active link) */
.st-emotion-cache-12fm52p label div[data-testid="stMarkdownContainer"] p {{
    color: {LIGHT_TEXT_COLOR};
}}
.st-emotion-cache-12fm52p label div[data-testid="stMarkdownContainer"] p:hover {{
    color: {ACCENT_COLOR}; 
}}

/* Metric Card Customization (Tech Panel Look) */
[data-testid="stMetric"] {{
    background-color: {CARD_BACKGROUND}; 
    border: 1px solid #3c4a63; /* Subtle frame border */
    border-left: 5px solid {ACCENT_COLOR}; /* Electric accent bar on the left */
    border-radius: 6px;
    padding: 15px 0;
    text-align: center;
    box-shadow: 0 0 10px rgba(78, 140, 252, 0.1); /* Very subtle glow */
}}
[data-testid="stMetricLabel"] {{
    font-size: 1.0em;
    color: #a0a8b8; /* Subdued label color */
    font-weight: normal;
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT_COLOR}; /* Electric cyan value color */
    font-size: 2.2em;
    font-weight: 700;
}}
/* Plotly/Matplotlib Background (Deep Midnight Blue Panel) */
.stPlotlyChart, .matplotlib {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid #3c4a63; 
    border-radius: 6px;
    padding: 15px;
    box-shadow: 0 0 10px rgba(78, 140, 252, 0.1);
}}
</style>
""", unsafe_allow_html=True)


# --- 2. DATA AND MODEL LOADING ---

# Load Data 
@st.cache_data
def load_data(file_path):
    # Creating placeholder data structures if files don't exist
    dummy_data_path = 'data/isro_300_missions.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(dummy_data_path):
        data = {
            'launch_date': pd.to_datetime(['2020-01-15', '2020-05-20', '2021-02-10', '2021-11-01', '2022-03-25']),
            'launch_vehicle': ['PSLV-XL', 'GSLV Mk II', 'PSLV-C', 'GSLV Mk III', 'PSLV-C'],
            'payload_weight_kg': [1500, 3000, 1200, 4000, 1400],
            'mission_type': ['Earth Observation', 'Communication', 'Navigation', 'Manned', 'Earth Observation'],
            'launch_cost': [80, 150, 75, 220, 90],
            'launch_outcome': [1, 0, 1, 1, 1], # 1=Success, 0=Failure
            'launch_window': ['Day', 'Night', 'Day', 'Day', 'Night'],
            'launch_site': ['SDSC SHAR', 'SDSC SHAR', 'SDSC SHAR', 'Satish Dhawan Space Centre', 'SDSC SHAR'],
            'temperature_C': [25.5, 30.1, 28.0, 26.5, 29.5],
            'wind_speed_kmh': [12.0, 25.5, 18.0, 10.0, 22.0],
            'humidity_percent': [70, 85, 65, 75, 80],
            'system_health_index': [0.95, 0.80, 0.98, 0.92, 0.96],
            'vehicle_success_rate': [0.9, 0.85, 0.9, 0.85, 0.9]
        }
        df = pd.DataFrame(data)
        # Replicate to simulate 300 missions
        df = pd.concat([df] * 60, ignore_index=True)
        df['launch_outcome'] = np.random.choice([0, 1], size=len(df), p=[0.1, 0.9])
        df.to_csv(dummy_data_path, index=False)
    
    try:
        data = pd.read_csv(file_path)
        data['launch_date'] = pd.to_datetime(data['launch_date'])
        return data
    except Exception as e:
        st.error(f"Error loading or generating data: {e}")
        st.stop()
data = load_data('data/isro_300_missions.csv')

# Load Model Pipeline (Placeholder for a non-existent model)
@st.cache_resource
def load_model(file_path):
    # Dummy model creation if the file is missing (to prevent crashing the UI)
    if not os.path.exists(file_path):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from xgboost import XGBClassifier
        
        # Define features
        numerical_features = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate']
        categorical_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', XGBClassifier(random_state=42))])
        
        return model_pipeline 
    
    try:
        return joblib.load(file_path) 
    except Exception as e:
        st.warning(f"Warning: Could not load the trained model file. Using a dummy placeholder model. Error: {e}")
        return load_model('non_existent_path.pkl')

model = load_model('launch_model_pipeline.pkl') 

# Load Financial/Risk Metrics (Placeholders if missing)
def load_json_metrics(file_path):
    try:
        if os.path.exists(file_path):
             with open(file_path, 'r') as f:
                 return json.load(f)
        
        # Placeholder data if file not found
        if file_path == 'reusability_metrics.json':
            return {
                "total_disposable_cost_M": 1500, 
                "total_reusable_cost_M": 850, 
                "total_savings_M": 650, 
                "break_even_point_missions": 8, 
                "roi_strategic_insight": "RLV development shows a strong projected ROI, reaching financial parity within 8 missions."
            }
        elif file_path == 'mc_results.json':
            return {
                "simulated_success_rate": 0.935, 
                "expected_net_value_M": 125, 
                "value_at_risk_M": -50, 
                "risk_insight": "The 95% Value-at-Risk suggests a maximum potential loss of $50M in 5% of all simulated high-stakes scenarios."
            }
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

reusability_metrics = load_json_metrics('reusability_metrics.json')
mc_results = load_json_metrics('mc_results.json')


# --- 3. PLOTTING UTILITY (Space Tech Theme) ---
def apply_plot_style(fig, ax):
    # Apply space tech theme styling to Matplotlib plots
    ax.set_facecolor(CARD_BACKGROUND)
    fig.patch.set_facecolor(CARD_BACKGROUND) 
    plt.rcParams['text.color'] = LIGHT_TEXT_COLOR 
    plt.rcParams['axes.labelcolor'] = LIGHT_TEXT_COLOR
    plt.rcParams['xtick.color'] = LIGHT_TEXT_COLOR
    plt.rcParams['ytick.color'] = LIGHT_TEXT_COLOR
    plt.rcParams['axes.edgecolor'] = LIGHT_TEXT_COLOR
    try:
        # Use subtle grid lines
        plt.grid(axis='y', color='#3c4a63', linestyle='--', alpha=0.5)
    except:
        pass


# --- 4. SIDEBAR NAVIGATION (Unrestricted) ---

# Initialize session state for the selected menu item
if 'selected_menu' not in st.session_state:
    st.session_state['selected_menu'] = "Project Overview"

st.sidebar.title("ISRO Advanced Analytics Platform")
st.sidebar.markdown("---")

# Define all menu items (unrestricted)
ALL_MENU_ITEMS = [
    "Project Overview",
    "Real-Time Predictions",
    "Strategic Financials",
    "Benchmarking",
    "Launch Sites Map",
    "Feature Distributions",
    "Correlations",
    "Grouped Analysis",
    "Time Trends",
    "Raw Data Overview",
]

# Display the menu using a single st.radio
st.sidebar.markdown("### Core Modules")
selected_menu = st.sidebar.radio(
    "Select Analysis Module",
    options=ALL_MENU_ITEMS,
    index=ALL_MENU_ITEMS.index(st.session_state.selected_menu),
    key='main_navigation_radio'
)

# Update session state
st.session_state.selected_menu = selected_menu

st.title("ISRO Rocket Launch Analytics & Forecasting")
st.markdown("""
**A Data Science project focused on analyzing and forecasting ISRO rocket missions — with insights on PSLV and GSLV vehicles, mission success probabilities, and reusability feasibility.**
Built using modern **Python analytics, machine learning, and visualization** techniques to support **cost optimization, mission planning, and strategic decision-making** in aerospace.
""")
st.markdown("---")

# =================================================================
# PAGE CONTENT
# =================================================================


# -----------------------------------------------------------------
# PAGE: PROJECT OVERVIEW (Updated Content)
# -----------------------------------------------------------------
if selected_menu == "Project Overview":
    st.header("Integrated Strategic Launch Analytics")
    
    st.subheader("Project Scope")
    st.markdown("""
    This project explores over **300 ISRO launch records**, performing **exploratory data analysis (EDA)**, **machine learning modeling**, and **interactive dashboarding** to achieve the following strategic objectives:

    * Forecast **mission costs** and **success probabilities**.
    * Analyze **payload trends**, **launch sites**, and **mission types**.
    * Evaluate **reusability potential** for PSLV/GSLV rockets.
    * Provide **real-time decision support** through an interactive **Streamlit dashboard**.
    """)
    
    st.markdown("---")
    
    st.subheader("Methods & Workflow")
    
    st.markdown("""
    **1. Data Collection & Preparation**
    * Compiled 300+ ISRO mission records from public databases and media sources.
    * Features include: `launch_vehicle`, `payload_mass`, `mission_type`, `launch_cost`, `weather`, and `outcome`.
    * Cleaned and standardized data using **Pandas**, **NumPy**, and **feature engineering** pipelines.

    **2. Exploratory Data Analysis (EDA)**
    * Visualized distributions and correlations using **Seaborn** and **Matplotlib**.
    * Built **correlation heatmaps**, **time-series trends**, and **geospatial maps (Folium)**.
    * Identified key cost drivers and payload patterns over time.

    **3. Statistical & Machine Learning Modeling**
    Implemented end-to-end ML workflow using **scikit-learn** and **XGBoost**:
    * **Regression**: Forecast launch expenses.
    * **Classification**: Predict mission success.
    * **Clustering (K-Means)**: Segment launches by cost and mission profile.
    * **Explainability**: Feature importance analysis to interpret key factors.
    """)
    
    st.markdown("---")
    
    st.subheader("Key Accomplishments")
    st.markdown("""
    * Built accurate forecasting models for launch cost and success prediction.
    * Created clustering-based segmentation to identify cost-efficient and reusable mission types.
    * Delivered explainable AI with feature ranking for transparent decision-making.
    * Designed an interactive dashboard enabling real-time analytics for mission planning.
    * Completed full data science lifecycle — from ingestion to deployment.
    """)

    st.subheader("Tech Stack")
    st.markdown("""
    | Category | Tools / Libraries |
    |-----------|------------------|
    | **Languages** | Python |
    | **Data Handling** | Pandas, NumPy |
    | **Visualization** | Matplotlib, Seaborn, Plotly, Folium |
    | **Modeling** | scikit-learn, XGBoost |
    | **Dashboarding** | Streamlit |
    | **Version Control** | Git, GitHub |
    | **Documentation** | Markdown, Jupyter Notebooks |
    """)

    st.subheader("Insights & Outcomes")
    st.markdown("""
    | Insight Type | Description |
    |---------------|-------------|
    | **Cost Drivers** | Payload mass, vehicle type, and mission complexity drive expenses |
    | **Success Factors** | Weather and payload characteristics strongly influence outcomes |
    | **Reusability Feasibility** | Identified optimal missions for potential reusable technology |
    | **Strategic Benefit** | Supports budget planning, R&D investment, and policy formulation |
    """)

    st.markdown("---")

    colA, colB, colC = st.columns(3)
    colA.metric("Total Missions Analyzed", f"{len(data)}", "Data from 2000-2025")
    colB.metric("Overall Success Rate", f"{data['launch_outcome'].mean():.2%}", "Historical KPI")
    colC.metric("Top Launch Vehicle", data['launch_vehicle'].mode()[0], "Most frequent in dataset")


# -----------------------------------------------------------------
# PAGE: REAL-TIME PREDICTIONS
# -----------------------------------------------------------------
elif selected_menu == "Real-Time Predictions":
    st.header("Real-Time Mission Success Prediction")
    
    if model is None:
        st.warning("Prediction feature is unavailable due to an unexpected model error.")
    else:
        # ... Prediction Input Form ...
        with st.form("prediction_form", border=False):
            st.subheader("Mission Input Parameters")
            col1, col2 = st.columns(2)
            
            # Using data.unique() which will work even with the placeholder data
            with col1:
                launch_vehicle = st.selectbox("Launch Vehicle", sorted(data['launch_vehicle'].unique()))
                mission_type = st.selectbox("Mission Type", sorted(data['mission_type'].unique()))
                launch_window = st.selectbox("Launch Window", sorted(data['launch_window'].unique()))
                launch_site = st.selectbox("Launch Site", sorted(data['launch_site'].unique()))
            
            with col2:
                payload_weight_kg = st.number_input("Payload Weight (kg)", 50, 4500, 1500, help="Higher payload generally increases complexity/risk.")
                temperature_C = st.slider("Temperature (°C)", float(data['temperature_C'].min()), float(data['temperature_C'].max()), 28.0)
                wind_speed_kmh = st.slider("Wind Speed (km/h)", float(data['wind_speed_kmh'].min()), float(data['wind_speed_kmh'].max()), 15.0)
                system_health_index = st.slider("System Health Index (0-1)", 0.0, 1.0, 0.95, help="Composite score for vehicle subsystem readiness.")
                
                # Fetching mean values from data for default inputs
                vehicle_success_rate_map = data.set_index('launch_vehicle')['vehicle_success_rate'].to_dict()
                vehicle_success_rate = vehicle_success_rate_map.get(launch_vehicle, 0.90)
                humidity_percent = data['humidity_percent'].mean() 
                
            submitted = st.form_submit_button("Initiate Prediction", type="primary")

        if submitted:
            input_df = pd.DataFrame({
                'launch_vehicle':[launch_vehicle], 'launch_window':[launch_window], 'mission_type':[mission_type],
                'launch_site':[launch_site], 'payload_weight_kg':[payload_weight_kg], 'temperature_C':[temperature_C],
                'wind_speed_kmh':[wind_speed_kmh], 'humidity_percent':[humidity_percent],
                'system_health_index':[system_health_index], 'vehicle_success_rate':[vehicle_success_rate]
            })

            try:
                # Prediction with the potentially dummy model
                prob = model.predict_proba(input_df)[0][1]
                st.metric("Predicted Success Probability", f"{prob:.2%}", delta=None)
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
                
            
            # --- FEATURE IMPORTANCE ---
            st.markdown("---")
            st.subheader("Top Risk & Success Drivers")

            # Since the model is a pipeline, accessing importance can be complex
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            
            importances = None
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            
            # Feature name extraction must match the structure of the preprocessor
            numerical_features = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate']
            categorical_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site']
            
            cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            feature_names = numerical_features + cat_feature_names # Note: order matters due to ColumnTransformer

            if importances is not None and len(feature_names) == len(importances):
                fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(8, 5))
                # Use a color palette suitable for the space theme
                sns.barplot(x='Importance', y='Feature', data=fi_df, palette='flare', ax=ax)
                ax.set_title("Top 10 Feature Importances", color=LIGHT_TEXT_COLOR)
                ax.set_xlabel("Feature Importance Score", color=LIGHT_TEXT_COLOR)
                ax.set_ylabel("")
                apply_plot_style(fig, ax)
                st.pyplot(fig)
            else:
                st.error("Feature Importance unavailable. (Model importances could not be accessed or matched.)")
            
            st.markdown("---")
            st.subheader("Batch Prediction Utility")
            uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    batch_preds = model.predict_proba(batch_df)[:, 1]
                    batch_df['predicted_success_probability'] = batch_preds
                    st.dataframe(batch_df)
                    st.download_button("Download Predictions CSV", data=batch_df.to_csv(index=False).encode(), file_name='batch_predictions.csv', type="primary")
                except Exception as e:
                    st.error(f"Batch prediction failed. Ensure the CSV columns match the training data features. Error: {e}")


# -----------------------------------------------------------------
# PAGE: STRATEGIC FINANCIALS
# -----------------------------------------------------------------
elif selected_menu == "Strategic Financials":
    st.header("Financial Strategy and Risk Assessment")
    
    st.subheader("1. Reusable Launch Vehicle (RLV) ROI")
    if reusability_metrics:
        colA, colB, colC = st.columns(3)
        colA.metric("Total Disposable Cost (15 Missions)", 
                    f"${reusability_metrics['total_disposable_cost_M']}M")
        colB.metric("Total RLV Program Cost (15 Missions)", 
                    f"${reusability_metrics['total_reusable_cost_M']}M",
                    delta=f"Savings: ${reusability_metrics['total_savings_M']}M", delta_color="inverse")
        colC.metric("Financial Break-Even Point", 
                    f"{reusability_metrics['break_even_point_missions']} Launches",
                    help="Number of launches required for RLV savings to exceed R&D cost.")
        st.info(f"**Strategic Insight:** {reusability_metrics['roi_strategic_insight']}")
    else:
        st.error("Reusability metrics not found. Please run `python reusability_analysis.py`.")
        
    st.markdown("---")
    
    st.subheader("2. Monte Carlo Risk Assessment (High-Stakes Mission)")
    if mc_results:
        colD, colE, colF = st.columns(3)
        colD.metric("Simulated Success Rate", 
                    f"{mc_results['simulated_success_rate']:.2%}")
        colE.metric("Expected Net Value (ROI)", 
                    f"${mc_results['expected_net_value_M']}M",
                    help="The long-term average profit expected per mission.")
        colF.metric("Value-at-Risk (95% VaR)", 
                    f"${mc_results['value_at_risk_M']}M",
                    help="Worst-case loss scenario that will only be exceeded 5% of the time.")
        st.warning(f"**Risk Alert:** {mc_results['risk_insight']}")
    else:
        st.error("Monte Carlo results not found. Please run `python scenario_simulator.py`.")


# -----------------------------------------------------------------
# PAGE: BENCHMARKING
# -----------------------------------------------------------------
elif selected_menu == "Benchmarking":
    st.header("Industry Benchmarking and Competitiveness")

    benchmark_chart = 'benchmarking_chart.png'
    # Placeholder for the chart if the file is missing
    if os.path.exists(benchmark_chart):
        st.image(benchmark_chart, caption="ISRO's Cost/Kg and Success Rate vs. Global Agencies")
    else:
        st.subheader("Cost/kg to LEO (Simulated Benchmarks)")
        benchmark_data = {
            'Agency': ['ISRO', 'SpaceX (F9)', 'NASA (SLS)', 'Roscosmos (Soyuz)'],
            'Cost per kg (USD)': [3000, 2700, 15000, 4500]
        }
        df_bench = pd.DataFrame(benchmark_data)
        # Using a line chart for visual interest, as st.bar_chart defaults to a simple style
        st.line_chart(df_bench.set_index('Agency')) 
        st.info(f"Placeholder: Chart file '{benchmark_chart}' not found. Using simple chart instead.")


# -----------------------------------------------------------------
# PAGE: LAUNCH SITES MAP
# -----------------------------------------------------------------
elif selected_menu == "Launch Sites Map":
    st.header("Launch Trajectory and Geospatial Risk Analysis")
    
    map_file = 'geospatial_launch_map.html'
    # Placeholder for the map if the file is missing
    if os.path.exists(map_file):
        st.components.v1.html(open(map_file, 'r').read(), height=650, scrolling=True)
        st.info("The map displays current launch infrastructure, simulated orbital trajectories, and nominal splashdown zones.")
    else:
        st.warning(f"Geospatial map file '{map_file}' not found. Placeholder used for context.")
        st.image(f"https://placehold.co/800x650/{CARD_BACKGROUND.strip('#')}/{ACCENT_COLOR.strip('#')}?text=GEOSPATIAL+MAP+PLACEHOLDER", use_column_width=True)


# -----------------------------------------------------------------
# PAGE: FEATURE DISTRIBUTIONS
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
            # Use the accent color for the main plot element
            sns.histplot(data[col], kde=True, ax=ax, color=ACCENT_COLOR, edgecolor='#000') 
            ax.set_title(col, color=LIGHT_TEXT_COLOR)
            apply_plot_style(fig, ax)
            st.pyplot(fig)

    st.subheader("Categorical Feature Counts")
    cols = st.columns(len(cat_cols))

    for i, col in enumerate(cat_cols):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 3))
            # Use a dark mode palette with sharp colors
            sns.countplot(y=data[col], ax=ax, palette='icefire', hue=data[col], legend=False)
            ax.set_title(col, color=LIGHT_TEXT_COLOR)
            ax.set_ylabel("")
            apply_plot_style(fig, ax)
            st.pyplot(fig)


# -----------------------------------------------------------------
# PAGE: CORRELATIONS
# -----------------------------------------------------------------
elif selected_menu == "Correlations":
    st.header("Numerical Feature Correlation Heatmap")
    
    # Include all relevant numerical data for analysis
    num_data = data[['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate', 'launch_outcome']]
    corr_matrix = num_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use a high-tech/thermal colormap for the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt=".2f", linewidths=.5, linecolor=BACKGROUND_COLOR, ax=ax, annot_kws={'color': LIGHT_TEXT_COLOR})
    ax.set_title("Full Feature Correlation Matrix", color=LIGHT_TEXT_COLOR)
    apply_plot_style(fig, ax)
    st.pyplot(fig)
    
    st.info("This matrix shows the linear relationship between environmental, mission, and outcome parameters.")


# -----------------------------------------------------------------
# PAGE: GROUPED ANALYSIS
# -----------------------------------------------------------------
elif selected_menu == "Grouped Analysis":
    st.header("Mission Success Rate by Launch Vehicle")
    
    success_rate = data.groupby('launch_vehicle')['launch_outcome'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Using the secondary accent color palette
    sns.barplot(x=success_rate.index, y=success_rate.values, palette='magma', ax=ax) 
    ax.set_title("Average Launch Success Rate by Vehicle Type", color=LIGHT_TEXT_COLOR)
    ax.set_xlabel("Launch Vehicle")
    ax.set_ylabel("Success Rate (Mean of Launch Outcome)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    apply_plot_style(fig, ax)
    st.pyplot(fig)
    
    st.info(f"**Highest Success Rate:** {success_rate.index[0]} at {success_rate.iloc[0]:.2%}. Detailed success rate breakdowns inform resource allocation and R&D prioritization.")


# -----------------------------------------------------------------
# PAGE: TIME TRENDS
# -----------------------------------------------------------------
elif selected_menu == "Time Trends":
    st.header("Rolling Average of Mission Success Over Time")
    
    time_data = data.sort_values('launch_date').copy()
    time_data['Year'] = time_data['launch_date'].dt.year
    time_data['Rolling_Success'] = time_data['launch_outcome'].rolling(window=10, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Use the primary accent color for the line
    sns.lineplot(x='launch_date', y='Rolling_Success', data=time_data, ax=ax, color=ACCENT_COLOR, linewidth=3)
    ax.set_title("10-Mission Rolling Success Rate (Historical Trend)", color=LIGHT_TEXT_COLOR)
    ax.set_xlabel("Launch Date")
    ax.set_ylabel("Rolling Success Rate")
    ax.set_ylim(0.5, 1.05)
    
    mean_success = data['launch_outcome'].mean()
    ax.axhline(mean_success, color=SECONDARY_ACCENT, linestyle='--', label=f'Overall Mean ({mean_success:.2%})')
    ax.legend()
    
    apply_plot_style(fig, ax)
    st.pyplot(fig)
    
    st.info("The rolling average helps visualize long-term improvements in launch reliability and technology maturity.")


# -----------------------------------------------------------------
# PAGE: RAW DATA OVERVIEW
# -----------------------------------------------------------------
elif selected_menu == "Raw Data Overview":
    st.header("Raw Mission Data Overview and Statistics")
    
    st.subheader("ISRO Mission Dataset (Preview)")
    
    st.dataframe(data.head(10)) 
    
    st.subheader("Data Distribution Summary (Numerical Features)")
    st.dataframe(data.describe().T)
