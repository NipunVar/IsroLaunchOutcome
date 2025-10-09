import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="ISRO Launch Success Predictor", page_icon=":rocket:", layout="wide")

# Modern deep-dark neon CSS styling and elegant fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Open+Sans&family=Montserrat:wght@700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp { 
    font-family: 'Open Sans', sans-serif; 
    background: #0a0a23 !important;
    color: #dcdcdc !important;
}
.stSidebar {
    background: linear-gradient(120deg, #16163d 80%, #6135a1 120%);
    border-right: 2px solid #ad76ff !important;
}
.stSidebar .sidebar-content {
    padding-top: 2rem;
}
h1 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 900;
    font-size: 3rem;
    text-align: center;
    background: linear-gradient(135deg, #bd68f7, #d289ff, #cfaafd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.4em;
    margin-bottom: 0.5em;
    letter-spacing: 2px;
}
h2 {
    font-family: 'Montserrat', sans-serif;
    color: #CDAAFF;
    font-weight: 700;
    margin-bottom: 1rem;
}
.overview-card {
    background: rgba(34, 31, 56, 0.95);
    border-left: 7px solid #a75afd;
    border-radius: 18px;
    padding: 2.1rem 3rem 2rem 3rem;
    margin: 2.2rem auto;
    line-height: 1.62;
    font-size: 1.10rem;
    max-width:900px;
    color: #ede3ff;
    box-shadow: 0 7px 32px 0 #31234285;
}
.overview-card p {text-align: justify;}
.metric-container {
    display: flex;
    justify-content: space-around;
    margin-top: 1.8rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #18102e;
    border-radius: 12px;
    padding: 1.2rem 2.3rem;
    min-width: 175px;
    font-weight: 800;
    font-size: 1.23rem;
    box-shadow: 0 0 15px #b176ff;
    color: #e1c3ff;
    text-align: center;
    border-left: 5px solid #a75afd;
}
.stMetric {
    background: #1a1334 !important;
    border-radius: 18px !important;
    color: #baa5ff !important;
    box-shadow: 0 3px 12px #8a75ff59 !important;
}
.stButton > button {
    background-image: linear-gradient(145deg, #b98cfa 20%, #ae4bfb 85%);
    border-radius: 10px;
    color: white;
    font-weight: 700;
    padding: 0.55rem 2rem;
    transition: 0.3s ease all;
}
.stButton > button:hover {
    background-image: linear-gradient(135deg, #ae4bfb 0%, #b98cfa 85%);
    color: #fff8fa;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚀 ISRO Launch Analytics")
menu = st.sidebar.radio("Choose Dashboard", [
    "Project Overview",
    "Raw Data Overview",
    "Feature Distributions",
    "Correlations",
    "Grouped Analysis",
    "Time Trends",
    "Launch Sites Map",
    "Prediction"
])

data = pd.read_csv('data/isro_300_missions.csv')

if menu == "Project Overview":
    st.markdown("<h1>🚀 ISRO Launch Analytics & Success Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='overview-card'>
    <p>
    <span style="font-size:1.21rem;color:#b887ff;font-weight:800;">
    Project Description
    </span><br><br>
    The ISRO Launch Analytics & Success Predictor is an advanced, modern data application focused on Indian spaceflight missions. It empowers researchers, engineers, and space enthusiasts to analyze, visualize, and predict rocket launch outcomes using a combination of highly interactive dashboards and machine learning.  
    <br><br>
    At its core, this project leverages over 300 mission records to reveal important patterns in launch vehicles, environmental variables, system health indices, and temporal trends. The modular dashboard layout brings together rich data exploration—via raw views, distributions, correlations, and launch site geospatial mapping—in a single, deeply dark-themed and visually sophisticated web interface.
    <br><br>
    The application goes beyond analytics: it offers a live AI-driven prediction tool that enables users to input mission parameters and instantly simulate the probability of mission success. Feature importance charts and batch CSV forecasting offer further explainability and flexibility for experts and learners alike.
    <br><br>
    The entire user experience is elevated by crisp typography, judicious use of color and gradients, and a card-based layout that emphasizes clarity and accessibility. Every metric, chart, and narrative section is designed to tell the story of ISRO's technological journey in a way that is not only beautiful but highly functional.
    <br><br>
    Ultimately, this project demonstrates the real impact of fusing aerospace domain knowledge with the best of modern data science and web technology, producing insights that are actionable, educational, and inspiring for the global space community.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Simple stats for the summary
    total_missions = len(data)
    total_successes = int(data['launch_outcome'].sum())
    total_failures = total_missions - total_successes
    site_counts = data['launch_site'].value_counts()

    # Chart summary, two columns: bar chart and pie chart side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 style='color:#a75afd;text-align:center'>Launch Site Usage</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3))
        site_counts.plot(kind='bar', color='#ba99ff', ax=ax)
        ax.set_ylabel("Number of Launches")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha='right', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#160e29')
        st.pyplot(fig)

    with col2:
        st.markdown("<h2 style='color:#a75afd;text-align:center'>Success vs Failure</h2>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.pie([total_successes, total_failures], labels=["Successes", "Failures"], autopct='%1.1f%%',
                colors=['#b987ff', '#ee6694'], startangle=145, wedgeprops=dict(alpha=0.85, width=0.6, edgecolor='#0a0a23'))
        ax2.axis('equal')
        st.pyplot(fig2)

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">🚀<br>Total Missions<br><span style="font-size:2rem;">{total_missions}</span></div>
        <div class="metric-card">🌙<br>Successes<br><span style="font-size:2rem; color:#a89eff;">{total_successes}</span></div>
        <div class="metric-card">🔥<br>Failures<br><span style="font-size:2rem; color:#ee6694;">{total_failures}</span></div>
    </div>
    """, unsafe_allow_html=True)

elif menu == "Raw Data Overview":
    st.markdown('<h2>📊 Raw Data & Metrics</h2>', unsafe_allow_html=True)
    st.dataframe(data.head(20), width=1000)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Missions", len(data))
    col2.metric("Total Successes", int(data['launch_outcome'].sum()))
    col3.metric("Total Failures", len(data) - int(data['launch_outcome'].sum()))

elif menu == "Feature Distributions":
    st.markdown('<h2>🎨 Feature Distributions</h2>', unsafe_allow_html=True)
    features = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate']
    colors = ['#ae74ff', '#3ac3db', '#ff5bae', '#ffe859', '#71e4c6', '#ff4cb2']
    for f, c in zip(features, colors):
        fig, ax = plt.subplots()
        sns.histplot(data[f], kde=True, ax=ax, color=c)
        ax.set_title(f"{f} Distribution", fontsize=14, color=c)
        st.pyplot(fig)

elif menu == "Correlations":
    st.markdown('<h2>🌠 Correlation Heatmap</h2>', unsafe_allow_html=True)
    features = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate']
    corr = data[features].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='bwr', ax=ax)
    st.pyplot(fig)

elif menu == "Grouped Analysis":
    st.markdown('<h2>🧩 Grouped Launch Outcomes</h2>', unsafe_allow_html=True)
    grouping = st.selectbox("Group by:", ['launch_vehicle', 'mission_type', 'launch_window', 'launch_site'])
    group_data = data.groupby(grouping)['launch_outcome'].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(group_data)
    st.write(group_data)

elif menu == "Time Trends":
    st.markdown('<h2>🌌 Launch Trends Over Time</h2>', unsafe_allow_html=True)
    data['launch_date'] = pd.to_datetime(data['launch_date'], errors='coerce')
    data['launch_year'] = data['launch_date'].dt.year
    min_year, max_year = int(data['launch_year'].min()), int(data['launch_year'].max())
    year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    filtered = data[(data['launch_year'] >= year_range[0]) & (data['launch_year'] <= year_range[1])]
    success_rate = filtered.groupby('launch_year')['launch_outcome'].mean()
    launches = filtered.groupby('launch_year').size()
    fig, ax1 = plt.subplots()
    ax1.plot(success_rate.index, success_rate.values, color="#ff5bae", marker='o', linewidth=3)
    ax1.set_ylabel("Success Rate", color="#ff5bae")
    ax2 = ax1.twinx()
    ax2.bar(launches.index, launches.values, alpha=0.25, color="#3ac3db")
    ax2.set_ylabel("Launches", color="#3ac3db")
    st.pyplot(fig)

elif menu == "Launch Sites Map":
    st.markdown('<h2>🗺️ ISRO Launch Site Map</h2>', unsafe_allow_html=True)
    mapdata = data.dropna(subset=['launch_site_lat', 'launch_site_lon']).copy()
    mapdata = mapdata.rename(columns={'launch_site_lat': 'latitude', 'launch_site_lon': 'longitude'})
    st.map(mapdata[['latitude', 'longitude']])

elif menu == "Prediction":
    st.markdown('<h2>🚀 Predict Launch Success</h2>', unsafe_allow_html=True)
    model = joblib.load('launch_model_pipeline.pkl')
    col1, col2 = st.columns(2)
    with col1:
        launch_vehicle = st.selectbox("Launch Vehicle", sorted(data['launch_vehicle'].unique()))
        mission_type = st.selectbox("Mission Type", sorted(data['mission_type'].unique()))
        launch_window = st.selectbox("Launch Window", sorted(data['launch_window'].unique()))
        launch_site = st.selectbox("Launch Site", sorted(data['launch_site'].unique()))
    with col2:
        payload_weight_kg = st.number_input("Payload Weight (kg)", 50, 4500, 1500)
        temperature_C = st.slider("Temperature (°C)", float(data['temperature_C'].min()), float(data['temperature_C'].max()), float(data['temperature_C'].mean()))
        wind_speed_kmh = st.slider("Wind Speed (km/h)", float(data['wind_speed_kmh'].min()), float(data['wind_speed_kmh'].max()), float(data['wind_speed_kmh'].mean()))
        humidity_percent = st.slider("Humidity (%)", float(data['humidity_percent'].min()), float(data['humidity_percent'].max()), float(data['humidity_percent'].mean()))
        system_health_index = st.slider("System Health Index", 0.0, 1.0, 0.7)
        vehicle_success_rate_map = data.set_index('launch_vehicle')['vehicle_success_rate'].to_dict()
        vehicle_success_rate = vehicle_success_rate_map.get(launch_vehicle, 0.75)

    input_df = pd.DataFrame({
        'launch_vehicle':[launch_vehicle],
        'launch_window':[launch_window],
        'mission_type':[mission_type],
        'launch_site':[launch_site],
        'payload_weight_kg':[payload_weight_kg],
        'temperature_C':[temperature_C],
        'wind_speed_kmh':[wind_speed_kmh],
        'humidity_percent':[humidity_percent],
        'system_health_index':[system_health_index],
        'vehicle_success_rate':[vehicle_success_rate]
    })

    if st.button("🌠 Predict Success Probability"):
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"Predicted Launch Success Probability: {prob:.2%}")

        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        cat_features = ['launch_vehicle', 'launch_window', 'mission_type', 'launch_site']
        ohe = preprocessor.named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(cat_features))
        num_features = ['payload_weight_kg', 'temperature_C', 'wind_speed_kmh', 'humidity_percent', 'system_health_index', 'vehicle_success_rate']
        feature_names = cat_feature_names + num_features
        importances = classifier.feature_importances_

        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(fi_df['Feature'], fi_df['Importance'], color="#e940bb")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_preds = model.predict_proba(batch_df)[:, 1]
        batch_df['predicted_success_probability'] = batch_preds
        st.dataframe(batch_df)
        st.download_button("Download Predictions CSV", data=batch_df.to_csv(index=False).encode(), file_name='batch_predictions.csv')
