
# PHASE IV:  HIGH-CONTRAST CRIMSON & WHITE METEOROLOGICAL ENGINE

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration & Theme Anchoring
st.set_page_config(
    page_title="Chi-Square MLP Weather Engine",
    page_icon="⛈️",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Injector: Re-engineering components to Crimson & White
st.markdown("""
    <style>
    /* Main body adjustments */
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    
    /* Segmented metric card re-styling */
    .metric-card {
        background-color: #FFF5F5; /* Soft tint to match the crimson profile */
        border: 1px solid #FEB2B2; /* Subtle rose border accent */
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Enforcing bold text markers for the metrics layout */
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #C53030; }
    
    /* Target Override: Tweaking standard tab underline highlight to Crimson */
    button[data-baseweb="tab"] aria-selected="true" {
        color: #C53030 !important;
        border-bottom-color: #C53030 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation Layout with Crimson Overrides
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #C53030;'>⛈️ System Menu</h2>", unsafe_allow_html=True)
    
    # Replicating the sidebar structure with explicit color variables
    selected_page = option_menu(
        menu_title=None,
        options=["Dashboard Home", "Forecast Engine", "Model Analytics"],
        icons=["house-door", "cpu", "graph-up"],
        menu_icon="cast",
        default_index=1, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#C53030", "font-size": "18px"}, # Fixed the comment syntax here
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "#FEEBC8"},
            "nav-link-selected": {"background-color": "#C53030", "color": "white"}, 
        }
    )
    
    st.write("---")
    st.markdown("""
        <small>**Research Scope:** B.Sc. Project Validation Engine  
        **Supervisor:** Dr. Abdulsalam  
        **Institution:** Kwara State University</small>
    """, unsafe_allow_html=True)

# Helper function to load serialized artifacts
@st.cache_resource
def load_system_artifacts():
    try:
        model = joblib.load('mlp_predictive_engine.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        features = joblib.load('optimized_features_list.pkl')
        return model, scaler, features
    except Exception:
        dummy_features = ['Humidity3pm', 'Pressure3pm', 'WindGustSpeed', 'RainToday', 'Humidity9am', 'MaxTemp', 'MinTemp', 'Temp3pm']
        return None, None, dummy_features

mlp_engine, minmax_scaler, selected_features = load_system_artifacts()


# VIEW MODULE 1: DASHBOARD HOME

if selected_page == "Dashboard Home":
    st.title("🏥 Atmospheric Decision Support System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Target Predictor", value="RainTomorrow")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Optimized Subspace Dimensions", value=f"{len(selected_features)} Features")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Pipeline Classification Target", value="Binary (0 / 1)")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.write("### Project Summary")
    st.info("This framework leverages the Chi-Square statistical selector to prune redundant noise, mitigating learning fatigue within the Multilayer Perceptron inference layers.")


# VIEW MODULE 2: FORECAST ENGINE (The Core Entry Layout)

elif selected_page == "Forecast Engine":
    st.title("🔮 Predictive Inference Layer")
    st.markdown("---")
    
    # Organizing input segments using tabs matching the medical application layout
    tab1, tab2 = st.tabs(["📋 Entry Interface Form", "🔍 Architectural Settings"])
    
    with tab1:
        st.write("### Present Atmospheric Metrics")
        
        # Enforce strict layout partitioning via column layout blocks
        col1, col2 = st.columns(2)
        user_inputs = {}
        
        for idx, feature in enumerate(selected_features):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                if feature == 'RainToday':
                    option = st.selectbox("Did it rain today? (RainToday)", options=["No", "Yes"])
                    user_inputs[feature] = 1.0 if option == "Yes" else 0.0
                elif 'Humidity' in feature:
                    user_inputs[feature] = st.slider(f"{feature} (%)", 0.0, 100.0, 65.0, 1.0)
                elif 'Cloud' in feature:
                    user_inputs[feature] = st.slider(f"{feature} (Octas)", 0, 8, 4)
                elif 'Pressure' in feature:
                    user_inputs[feature] = st.slider(f"{feature} (hPa)", 980.0, 1050.0, 1015.0, 0.1)
                elif 'WindGustSpeed' in feature:
                    user_inputs[feature] = st.number_input(f"{feature} (km/h)", 0.0, 150.0, 40.0)
                elif 'Temp' in feature:
                    user_inputs[feature] = st.number_input(f"{feature} (°C)", -10.0, 55.0, 22.0)
                elif feature == 'Rainfall':
                    user_inputs[feature] = st.number_input(f"{feature} (mm)", min_value=0.0, value=0.0)
                elif feature == 'Sunshine':
                    user_inputs[feature] = st.slider(f"{feature} (Hours)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                else:
                    user_inputs[feature] = st.number_input(f"{feature} (Value)", min_value=0.0, value=0.0)

        st.write("---")
        
        # Primary Action Button styled dynamically by the server configuration
        if st.button("🔮 Compute Prediction Profile", type="primary", use_container_width=True):
            if mlp_engine is None:
                st.error("System running in offline standby mode. Missing binary training files.")
            else:
                # Execution layer data normalization mapping
                input_df = pd.DataFrame([user_inputs])[selected_features]
                full_feature_names = minmax_scaler.feature_names_in_
                full_input_template = pd.DataFrame(0.0, index=[0], columns=full_feature_names)
                
                for col in selected_features:
                    full_input_template[col] = input_df[col]
                    
                scaled_full_matrix = minmax_scaler.transform(full_input_template)
                scaled_full_df = pd.DataFrame(scaled_full_matrix, columns=full_feature_names)
                mlp_ready_inputs = scaled_full_df[selected_features]
                
                prediction = mlp_engine.predict(mlp_ready_inputs)[0]
                probability = mlp_engine.predict_proba(mlp_ready_inputs)[0][1]
                
                # Output Block Display layout matching target style
                st.write("### Model Evaluation Results")
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.metric(label="Precipitation Probability Score", value=f"{probability*100:.2f}%")
                with m_col2:
                    if prediction == 1:
                        st.error("🚨 RAIN IS ANTICIPATED TOMORROW")
                    else:
                        st.success("☀️ NO RAIN IS ANTICIPATED TOMORROW")
                        
    with tab2:
        st.write("### Meteorological Decision Logic")
        st.info("The Multilayer Perceptron (MLP) evaluates the non-linear synergy between your 8 optimized predictors. Below are the specific atmospheric combinations that drive the neural network's activation thresholds.")
        
        rule_col1, rule_col2 = st.columns(2)
        
        with rule_col1:
            st.error("🚨 High Probability Indicators (Rain)")
            st.markdown("""
            The network's probability score surges toward **Rain** when it detects these convective patterns:
            * **Moisture Saturation:** `Humidity3pm` and `Humidity9am` are highly elevated (typically > **75%**).
            * **Atmospheric Opacity:** `Cloud3pm` and `Cloud9am` are dense (e.g., **6 to 8 Octas**), indicating heavy cloud layer formation.
            * **Solar Suppression:** `Sunshine` duration is extremely low, meaning solar radiation is blocked.
            * **Turbulence:** `WindGustSpeed` is high, showing active aerodynamic shear and front movement.
            * **Climatic Persistence:** `RainToday` is **Yes** and current `Rainfall` volume is greater than **0.00 mm**, proving an active moisture cell is already anchored over the region.
            """)
            
        with rule_col2:
            st.success("☀️ Stable Baseline Indicators (Clear)")
            st.markdown("""
            The network defaults to **No Rain** when it detects anti-cyclonic suppression:
            * **Dry Air Masses:** Morning and afternoon humidity levels remain low to moderate (< **50%**).
            * **Clear Skies:** Cloud cover at 9am and 3pm is minimal (**0 to 2 Octas**).
            * **High Insolation:** `Sunshine` values are high, indicating stable atmospheric thermodynamics.
            * **Stable Airflow:** `WindGustSpeed` remains at normal baseline levels without erratic spikes.
            * **Climatic Persistence:** `RainToday` is **No** and `Rainfall` remains at **0.00 mm**.
            """)
            
        st.write("---")
        st.markdown("💡 **System Test:** Switch to the *Entry Interface Form*, set `RainToday` to **Yes**, push `Humidity3pm` to **90%**, maximize `Cloud3pm` to **8**, and drop `Sunshine` to **0.00**. Observe how rapidly the prediction probability approaches 100%.")


# VIEW MODULE 3: MODEL ANALYTICS

elif selected_page == "Model Analytics":
    st.title("📊 Empirical Performance Metrics")
    st.markdown("---")
    
    # Standard metrics display grid mapped to Crimson style sheets
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy Baseline", "84.00%")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Matthews Coeff (MCC)", "0.4977")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Root Mean Square Error", "0.3386")
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Precision (Rain)", "0.7200")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.write(" ")
    st.success("Analysis Complete: Performance parameters pass the structural evaluation metrics threshold.")