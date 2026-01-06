# main.py
# Streamlit Dashboard for Ocean Pollution Prediction
# Run with: streamlit run main.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime, timedelta
import time
import sys
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ocean Pollution Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ocean-pollution-dashboard',
        'Report a bug': "https://github.com/ocean-pollution-dashboard/issues",
        'About': "# 🌊 Ocean Pollution Monitoring Dashboard"
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    
    /* Prediction cards */
    .prediction-card {
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 6px solid;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    .low-pollution {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left-color: #4CAF50;
    }
    
    .medium-pollution {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left-color: #FF9800;
    }
    
    .high-pollution {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left-color: #F44336;
    }
    
    /* Metric cards */
    .metric-card {
        padding: 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        text-align: center;
        border: 2px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #1E88E5;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    .status-online { 
        background-color: #4CAF50;
        box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
    }
    
    .status-offline { 
        background-color: #F44336;
        box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7);
    }
    
    .status-warning { 
        background-color: #FF9800;
        box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.7);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    /* Custom button styles */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1E88E5, #0D47A1);
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e3f2fd;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
class APIConfig:
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 10
    POLLUTION_LEVELS = ["LOW", "MEDIUM", "HIGH"]
    LEVEL_COLORS = {
        "LOW": "#4CAF50",
        "MEDIUM": "#FF9800",
        "HIGH": "#F44336"
    }
    LEVEL_EMOJIS = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🚨"
    }

class APIClient:
    """Client for interacting with the Ocean Pollution API"""
    
    @staticmethod
    def get_health():
        """Check API health"""
        try:
            response = requests.get(
                f"{APIConfig.BASE_URL}/health",
                timeout=APIConfig.TIMEOUT
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def get_model_info():
        """Get model information"""
        try:
            response = requests.get(
                f"{APIConfig.BASE_URL}/model-info",
                timeout=APIConfig.TIMEOUT
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def predict_single(data: Dict[str, Any]):
        """Make single prediction"""
        try:
            response = requests.post(
                f"{APIConfig.BASE_URL}/predict",
                json=data,
                timeout=APIConfig.TIMEOUT
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def predict_batch(data: Dict[str, Any]):
        """Make batch prediction"""
        try:
            response = requests.post(
                f"{APIConfig.BASE_URL}/predict/batch",
                json=data,
                timeout=APIConfig.TIMEOUT * 2
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_project_structure():
        """Get project structure"""
        try:
            response = requests.get(
                f"{APIConfig.BASE_URL}/project/structure",
                timeout=APIConfig.TIMEOUT
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def get_sample_prediction():
        """Get sample prediction"""
        try:
            response = requests.get(
                f"{APIConfig.BASE_URL}/predict/sample",
                timeout=APIConfig.TIMEOUT
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None

# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    
    if 'api_health' not in st.session_state:
        st.session_state.api_health = None
    
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Real-time Prediction"
    
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = {
            "CHL": 2.5,
            "PP": 350.0,
            "KD490": 0.25,
            "DIATO": 0.03,
            "DINO": 0.02
        }

# Sidebar component
def render_sidebar():
    """Render sidebar with navigation and status"""
    with st.sidebar:
        # Logo and title
        st.image("https://cdn-icons-png.flaticon.com/512/3097/3097140.png", width=100)
        st.markdown('<h2 style="text-align: center;">🌊 Ocean Pollution Monitor</h2>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        nav_options = [
            "🏠 Dashboard",
            "🔬 Real-time Prediction", 
            "📊 Batch Analysis",
            "📈 Data Insights",
            "🤖 Model Information",
            "📋 Prediction History",
            "⚙️ Settings"
        ]
        
        selected_nav = st.radio(
            "Select Section:",
            nav_options,
            label_visibility="collapsed"
        )
        
        st.session_state.current_tab = selected_nav
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        # Check API health
        api_health = APIClient.get_health()
        st.session_state.api_health = api_health
        
        if api_health:
            status = api_health.get("status", "unknown")
            model_loaded = api_health.get("model_loaded", False)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if status == "healthy":
                    st.markdown('<div class="status-indicator status-online"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-indicator status-offline"></div>', unsafe_allow_html=True)
            
            with col2:
                st.write(f"**API:** {status.upper()}")
            
            if model_loaded:
                st.success("✅ ML Model Loaded")
            else:
                st.warning("⚠️ Rule-based Mode")
        else:
            st.error("❌ API Unavailable")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        with col_btn2:
            if st.button("📊 Sample", use_container_width=True):
                st.session_state.run_sample = True
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.info("""
        **Ocean Pollution Dashboard**
        
        Monitor and predict ocean water quality
        using machine learning models.
        
        Version: 2.0.0
        Last Updated: 2024
        """)

# Main dashboard component
def render_dashboard():
    """Render main dashboard view"""
    st.markdown('<h1 class="main-header">🌊 Ocean Pollution Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Statistics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">🌡️ Water Quality<br><span style="font-size: 1.5rem; font-weight: bold;">Real-time</span></div>', unsafe_allow_html=True)
    
    with col2:
        total_predictions = len(st.session_state.predictions_history)
        st.markdown(f'<div class="metric-card">📈 Predictions<br><span style="font-size: 1.5rem; font-weight: bold;">{total_predictions}</span></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.api_health:
            status = "Online" if st.session_state.api_health.get("status") == "healthy" else "Offline"
        else:
            status = "Offline"
        st.markdown(f'<div class="metric-card">🔗 API Status<br><span style="font-size: 1.5rem; font-weight: bold;">{status}</span></div>', unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f'<div class="metric-card">🕒 Last Update<br><span style="font-size: 1.5rem; font-weight: bold;">{current_time}</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick prediction section
    st.markdown('<h3 class="sub-header">⚡ Quick Prediction</h3>', unsafe_allow_html=True)
    
    col_q1, col_q2, col_q3 = st.columns([2, 1, 2])
    
    with col_q1:
        quick_chl = st.slider(
            "Chlorophyll (CHL) mg/m³",
            min_value=0.0,
            max_value=20.0,
            value=2.5,
            step=0.1,
            key="quick_chl"
        )
    
    with col_q2:
        st.write("")  # Spacer
        quick_predict = st.button("Predict Now", type="primary", use_container_width=True)
    
    with col_q3:
        if quick_predict:
            with st.spinner("Making prediction..."):
                result = APIClient.predict_single({"CHL": quick_chl})
                
                if result and result.get("success"):
                    prediction = result["prediction"]
                    level = prediction["level"]
                    
                    st.markdown(f"""
                    <div class="prediction-card {level.lower()}-pollution">
                        <h3 style="text-align: center; margin: 0;">
                            {APIConfig.LEVEL_EMOJIS.get(level, "📊")} {level}
                        </h3>
                        <p style="text-align: center; font-size: 0.9rem;">
                            Confidence: {prediction['confidence']:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.predictions_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "input": {"CHL": quick_chl},
                        "result": prediction
                    })
                else:
                    st.error("Prediction failed")
    
    st.markdown("---")
    
    # Recent predictions
    if st.session_state.predictions_history:
        st.markdown('<h3 class="sub-header">📋 Recent Predictions</h3>', unsafe_allow_html=True)
        
        # Show last 5 predictions
        recent = st.session_state.predictions_history[-5:]
        
        for pred in reversed(recent):
            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            
            with col_p1:
                time_str = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M")
                st.write(f"**Time:** {time_str}")
                st.write(f"**CHL:** {pred['input'].get('CHL', 'N/A')}")
            
            with col_p2:
                level = pred["result"]["level"]
                st.write(f"**Level:** {level}")
            
            with col_p3:
                confidence = pred["result"]["confidence"]
                st.write(f"**Confidence:** {confidence:.1%}")
            
            st.markdown("---")

# Real-time prediction component
def render_realtime_prediction():
    """Render real-time prediction interface"""
    st.markdown('<h1 class="main-header">🔬 Real-time Pollution Prediction</h1>', unsafe_allow_html=True)
    
    # Use tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📁 File Upload", "🧪 Sample Data"])
    
    with tab1:
        st.markdown("### Enter Ocean Parameters")
        
        # Create two columns for input layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🌱 Essential Parameters")
            chl = st.number_input(
                "**Chlorophyll (CHL) mg/m³** *",
                min_value=0.0,
                max_value=50.0,
                value=2.5,
                step=0.1,
                help="Primary indicator of algal biomass - REQUIRED",
                key="chl_input"
            )
            
            st.markdown("#### 🌊 Biological Parameters")
            diato = st.number_input(
                "Diatom Concentration (DIATO)",
                min_value=0.0,
                max_value=5.0,
                value=0.05,
                step=0.01,
                help="Diatom phytoplankton concentration"
            )
            
            dino = st.number_input(
                "Dinoflagellate Concentration (DINO)",
                min_value=0.0,
                max_value=5.0,
                value=0.03,
                step=0.01,
                help="Dinoflagellate phytoplankton concentration"
            )
        
        with col_right:
            st.markdown("#### 📊 Physical Parameters")
            pp = st.number_input(
                "Primary Production (PP)",
                min_value=0.0,
                max_value=2000.0,
                value=350.0,
                step=10.0,
                help="Primary production rate"
            )
            
            kd490 = st.number_input(
                "Diffuse Attenuation (KD490)",
                min_value=0.0,
                max_value=5.0,
                value=0.25,
                step=0.01,
                help="Diffuse attenuation coefficient at 490nm"
            )
            
            st.markdown("#### 🧪 Chemical Parameters")
            cdm = st.number_input(
                "Colored Dissolved Matter (CDM)",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.001,
                help="Colored dissolved organic matter"
            )
            
            bbp = st.number_input(
                "Particulate Backscattering (BBP)",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                help="Particulate backscattering coefficient"
            )
        
        # Prediction settings
        st.markdown("---")
        st.markdown("### ⚙️ Prediction Settings")
        
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            save_prediction = st.checkbox("Save to history", value=True)
            show_details = st.checkbox("Show detailed results", value=True)
        
        with col_set2:
            prediction_mode = st.radio(
                "Mode:",
                ["Single Prediction", "Test Multiple Values"],
                horizontal=True
            )
        
        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button(
                "🚀 Run Prediction Analysis",
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            # Prepare input data
            input_data = {
                "CHL": chl,
                "PP": pp if pp > 0 else None,
                "KD490": kd490 if kd490 > 0 else None,
                "DIATO": diato if diato > 0 else None,
                "DINO": dino if dino > 0 else None,
                "CDM": cdm if cdm > 0 else None,
                "BBP": bbp if bbp > 0 else None
            }
            
            # Remove None values
            input_data = {k: v for k, v in input_data.items() if v is not None}
            
            # Show input summary
            with st.expander("📥 Input Data Summary", expanded=True):
                df_input = pd.DataFrame([input_data])
                st.dataframe(df_input, use_container_width=True)
            
            # Make prediction
            with st.spinner("🔍 Analyzing ocean parameters..."):
                result = APIClient.predict_single(input_data)
            
            if result and result.get("success"):
                prediction = result["prediction"]
                level = prediction["level"]
                confidence = prediction["confidence"]
                
                # Display prediction result
                st.markdown("---")
                st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Result card
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown(f"""
                    <div class="prediction-card {level.lower()}-pollution">
                        <div style="text-align: center;">
                            <span style="font-size: 3rem;">
                                {APIConfig.LEVEL_EMOJIS.get(level, "📊")}
                            </span>
                            <h2 style="color: {APIConfig.LEVEL_COLORS.get(level, '#000')}; margin: 10px 0;">
                                {level}
                            </h2>
                            <p style="font-size: 1.2rem; font-weight: bold;">
                                Confidence: {confidence:.1%}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick metrics
                    st.metric("Chlorophyll Value", f"{chl} mg/m³")
                    st.metric("Features Used", prediction["features_used"])
                    st.metric("Model Type", result.get("model_info", {}).get("model_type", "N/A"))
                
                with col_res2:
                    # Probability visualization
                    st.markdown("#### 📊 Probability Distribution")
                    
                    probabilities = prediction["probabilities"]
                    
                    # Create bar chart
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=list(probabilities.keys()),
                            y=list(probabilities.values()),
                            marker_color=[APIConfig.LEVEL_COLORS.get(k, "#999") for k in probabilities.keys()],
                            text=[f'{v:.1%}' for v in probabilities.values()],
                            textposition='auto',
                            marker_line_color='black',
                            marker_line_width=1,
                            hovertemplate="<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>"
                        )
                    ])
                    
                    fig_bar.update_layout(
                        title="Pollution Level Probabilities",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig_bar, width='stretch', key="rt_bar_chart")
                    
                    # Create pie chart
                    fig_pie = px.pie(
                        values=list(probabilities.values()),
                        names=list(probabilities.keys()),
                        title="Probability Distribution",
                        color=list(probabilities.keys()),
                        color_discrete_map=APIConfig.LEVEL_COLORS,
                        hole=0.3
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>"
                    )
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        st.plotly_chart(fig_bar, width='stretch', key="rt_bar_chart_1")
                    with col_chart2:
                        st.plotly_chart(fig_pie, width='stretch', key="rt_pie_chart_1")
                
                # Recommendations
                st.markdown("---")
                st.markdown("### 📋 Recommendations & Actions")
                
                if level == "LOW":
                    st.success("""
                    ### ✅ **WATER QUALITY: EXCELLENT**
                    
                    **Recommended Actions:**
                    - Continue normal monitoring schedule
                    - Maintain current conservation practices
                    - Schedule next routine check in 4 weeks
                    
                    **Environmental Status:**
                    - Safe for all marine activities
                    - Optimal for marine life
                    - No immediate concerns
                    """)
                    
                    col_rec1, col_rec2 = st.columns(2)
                    with col_rec1:
                        st.info("""
                        **Monitoring Schedule:**
                        - Weekly sampling recommended
                        - Continue baseline measurements
                        - Document environmental conditions
                        """)
                    
                    with col_rec2:
                        st.info("""
                        **Best Practices:**
                        - Maintain sampling protocols
                        - Continue data collection
                        - Regular equipment checks
                        """)
                
                elif level == "MEDIUM":
                    st.warning("""
                    ### ⚠️ **WATER QUALITY: MODERATE CONCERN**
                    
                    **Recommended Actions:**
                    - Increase monitoring frequency to twice weekly
                    - Investigate potential pollution sources
                    - Implement preventive measures
                    - Notify local environmental authorities
                    
                    **Environmental Status:**
                    - Caution advised for sensitive activities
                    - Monitor for changes
                    - Potential risk to sensitive species
                    """)
                    
                    col_rec1, col_rec2 = st.columns(2)
                    with col_rec1:
                        st.info("""
                        **Immediate Actions:**
                        - Review recent activity logs
                        - Check weather patterns
                        - Verify sensor accuracy
                        """)
                    
                    with col_rec2:
                        st.info("""
                        **Preventive Measures:**
                        - Increase sampling points
                        - Test for specific contaminants
                        - Review industrial activities
                        """)
                
                else:  # HIGH
                    st.error("""
                    ### 🚨 **WATER QUALITY: HIGH ALERT**
                    
                    **Emergency Actions Required:**
                    1. **IMMEDIATELY** notify environmental authorities
                    2. Activate emergency response plan
                    3. Issue public health advisory
                    4. Restrict water activities
                    5. Deploy cleanup operations
                    
                    **Environmental Status:**
                    - **DANGEROUS** conditions
                    - High risk to marine life
                    - Public health threat
                    - Immediate action required
                    """)
                    
                    col_rec1, col_rec2 = st.columns(2)
                    with col_rec1:
                        st.info("""
                        **Emergency Contacts:**
                        - Environmental Agency: 123-456-7890
                        - Coast Guard: 987-654-3210
                        - Health Department: 555-123-4567
                        - Local Authorities: 111-222-3333
                        """)
                    
                    with col_rec2:
                        st.info("""
                        **Response Protocol:**
                        - Evacuate affected areas
                        - Set up monitoring stations
                        - Collect water samples
                        - Document all observations
                        """)
                
                # Save to history
                if save_prediction:
                    history_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "input": input_data,
                        "result": prediction,
                        "model_info": result.get("model_info", {})
                    }
                    st.session_state.predictions_history.append(history_entry)
                    
                    if len(st.session_state.predictions_history) > 100:
                        st.session_state.predictions_history = st.session_state.predictions_history[-100:]
                    
                    st.toast("✅ Prediction saved to history!", icon="✅")
                
                # Show detailed results if requested
                if show_details:
                    with st.expander("📄 Detailed Results", expanded=False):
                        st.json(result)
            
            else:
                st.error("❌ Prediction failed. Please check your input and API connection.")
                if result and "error" in result:
                    st.error(f"Error: {result['error']}")
    
    with tab2:
        st.markdown("### 📁 Upload Data File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with ocean data",
            type=['csv', 'xlsx', 'txt'],
            help="File should contain at least a 'CHL' column"
        )
        
        if uploaded_file:
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep='\t')
                
                st.success(f"✅ File loaded successfully: {uploaded_file.name}")
                st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                
                # Show preview
                with st.expander("📊 Data Preview", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Check for required columns
                if 'CHL' not in df.columns:
                    st.error("❌ CSV must contain 'CHL' column")
                else:
                    if st.button("🔍 Analyze All Samples", type="primary"):
                        with st.spinner("Processing data..."):
                            # Prepare batch request
                            samples = []
                            for _, row in df.iterrows():
                                sample = {}
                                for col in ['CHL', 'PP', 'KD490', 'DIATO', 'DINO', 'CDM', 'BBP']:
                                    if col in row:
                                        sample[col] = float(row[col]) if pd.notnull(row[col]) else None
                                samples.append(sample)
                            
                            # Make batch prediction
                            result = APIClient.predict_batch({"samples": samples[:50]})  # Limit to 50 samples
                            
                            if result and result.get("success"):
                                st.success(f"✅ Processed {len(result['results'])} samples")
                                
                                # Display results
                                results_data = []
                                for i, res in enumerate(result["results"]):
                                    if res["success"]:
                                        pred = res["prediction"]
                                        results_data.append({
                                            "Sample": i + 1,
                                            "CHL": samples[i].get("CHL", "N/A"),
                                            "Level": pred["level"],
                                            "Confidence": pred["confidence"],
                                            "Features": pred["features_used"]
                                        })
                                
                                if results_data:
                                    results_df = pd.DataFrame(results_data)
                                    
                                    # Display results table
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Statistics
                                    st.markdown("#### 📈 Analysis Statistics")
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        total = len(results_data)
                                        st.metric("Total Samples", total)
                                    with col_stat2:
                                        high_count = len([r for r in results_data if r["Level"] == "HIGH"])
                                        st.metric("High Risk", high_count)
                                    with col_stat3:
                                        avg_conf = results_df["Confidence"].mean()
                                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                                    
                                    # Visualization
                                    st.markdown("#### 📊 Distribution Visualization")
                                    
                                    col_viz1, col_viz2 = st.columns(2)
                                    with col_viz1:
                                        # Level distribution
                                        level_counts = results_df["Level"].value_counts()
                                        fig_levels = px.pie(
                                            values=level_counts.values,
                                            names=level_counts.index,
                                            title="Pollution Level Distribution",
                                            color=level_counts.index,
                                            color_discrete_map=APIConfig.LEVEL_COLORS
                                        )
                                        st.plotly_chart(fig_levels, width='stretch', key="file_levels_chart")
                                    
                                    with col_viz2:
                                        # Confidence distribution
                                        fig_conf = px.histogram(
                                            results_df,
                                            x="Confidence",
                                            nbins=20,
                                            title="Confidence Distribution",
                                            color_discrete_sequence=['#1E88E5']
                                        )
                                        st.plotly_chart(fig_conf, width='stretch', key="file_conf_chart")
                                    
                                    # Download results
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv,
                                        file_name="pollution_analysis_results.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.error("❌ Batch prediction failed")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.markdown("### 🧪 Sample Data & Testing")
        
        col_samp1, col_samp2 = st.columns(2)
        
        with col_samp1:
            st.markdown("#### Example Data Sets")
            
            example_type = st.selectbox(
                "Select Example Scenario:",
                ["Clear Ocean", "Coastal Waters", "Polluted Bay", "Algal Bloom", "Custom"]
            )
            
            # Pre-defined examples
            examples = {
                "Clear Ocean": {"CHL": 0.3, "PP": 150, "KD490": 0.1, "DIATO": 0.01},
                "Coastal Waters": {"CHL": 2.5, "PP": 350, "KD490": 0.25, "DIATO": 0.05},
                "Polluted Bay": {"CHL": 8.0, "PP": 800, "KD490": 0.5, "DIATO": 0.2},
                "Algal Bloom": {"CHL": 15.0, "PP": 1200, "KD490": 0.8, "DIATO": 0.5},
                "Custom": {}
            }
            
            if example_type != "Custom":
                example_data = examples[example_type]
                
                # Display example data
                st.markdown("**Example Parameters:**")
                for key, value in example_data.items():
                    st.write(f"- {key}: {value}")
                
                if st.button(f"Test {example_type} Scenario", type="secondary"):
                    # Update input fields
                    st.session_state.sample_data = example_data
                    st.rerun()
        
        with col_samp2:
            st.markdown("#### Quick Test")
            
            if st.button("Get Sample Prediction from API", icon="🧪"):
                with st.spinner("Getting sample prediction..."):
                    result = APIClient.get_sample_prediction()
                
                if result and result.get("success"):
                    st.success("✅ Sample prediction received")
                    
                    sample_data = result.get("sample_data", {})
                    pred_result = result.get("result", {}).get("prediction", {})
                    
                    # Display results
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown("**Sample Data:**")
                        for key, value in sample_data.items():
                            st.write(f"- {key}: {value}")
                    
                    with col_res2:
                        if pred_result:
                            st.markdown("**Prediction:**")
                            st.write(f"- Level: {pred_result.get('level', 'N/A')}")
                            st.write(f"- Confidence: {pred_result.get('confidence', 0):.1%}")
                            st.write(f"- CHL: {pred_result.get('chl_value', 'N/A')}")
        
        st.markdown("---")
        st.markdown("#### 🎯 Test Multiple Values")
        
        test_values = st.text_area(
            "Enter multiple CHL values (comma separated):",
            "0.5, 1.0, 2.5, 5.0, 10.0",
            help="Enter CHL values separated by commas"
        )
        
        if st.button("Run Multiple Tests", icon="🔬"):
            try:
                values = [float(x.strip()) for x in test_values.split(',')]
                
                progress_bar = st.progress(0)
                results = []
                
                for i, value in enumerate(values):
                    result = APIClient.predict_single({"CHL": value})
                    if result and result.get("success"):
                        pred = result["prediction"]
                        results.append({
                            "CHL": value,
                            "Level": pred["level"],
                            "Confidence": pred["confidence"],
                            "Color": APIConfig.LEVEL_COLORS.get(pred["level"], "#999")
                        })
                    
                    progress_bar.progress((i + 1) / len(values))
                
                progress_bar.empty()
                
                if results:
                    # Create visualization
                    df_test = pd.DataFrame(results)
                    
                    fig = px.scatter(
                        df_test,
                        x="CHL",
                        y="Confidence",
                        color="Level",
                        title="CHL vs Confidence",
                        color_discrete_map=APIConfig.LEVEL_COLORS,
                        size=[20] * len(df_test),
                        hover_data=["Level", "Confidence"]
                    )
                    
                    fig.update_layout(
                        xaxis_title="Chlorophyll (CHL) mg/m³",
                        yaxis_title="Confidence",
                        height=400
                    )
                    
                    st.plotly_chart(fig, width='stretch', key="test_scatter_chart")
                    
                    # Show results table
                    st.dataframe(df_test[["CHL", "Level", "Confidence"]], use_container_width=True)
            except Exception as e:
                st.error(f"Error in testing: {str(e)}")

# Batch analysis component
def render_batch_analysis():
    """Render batch analysis interface"""
    st.markdown('<h1 class="main-header">📊 Batch Data Analysis</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🗺️ Spatial Analysis", "📋 Statistical Analysis"])
    
    with tab1:
        st.markdown("### Time Series Analysis")
        
        # Generate sample time series data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
        chl_values = np.random.uniform(0.1, 12, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 2
        
        # Create sample dataframe
        ts_df = pd.DataFrame({
            'Date': dates,
            'CHL': chl_values,
            'PP': chl_values * 100 + np.random.normal(0, 20, len(dates)),
            'KD490': 0.1 + (chl_values * 0.05) + np.random.normal(0, 0.02, len(dates))
        })
        
        # Display time series
        fig_ts = px.line(ts_df, x='Date', y='CHL', title='Chlorophyll Concentration Over Time')
        st.plotly_chart(fig_ts, width='stretch', key="ts_line_chart")
        
        # Add prediction analysis
        if st.button("Analyze Time Series", type="primary"):
            with st.spinner("Analyzing time series data..."):
                # Sample predictions for chart
                sample_dates = dates[::30]  # Every 30 days
                sample_chl = chl_values[::30]
                
                predictions = []
                for chl in sample_chl:
                    # Simulate prediction based on CHL value
                    if chl < 1:
                        level = "LOW"
                    elif chl < 5:
                        level = "MEDIUM"
                    else:
                        level = "HIGH"
                    predictions.append(level)
                
                # Create prediction chart
                pred_df = pd.DataFrame({
                    'Date': sample_dates,
                    'CHL': sample_chl,
                    'Prediction': predictions
                })
                
                fig_pred = px.scatter(
                    pred_df,
                    x='Date',
                    y='CHL',
                    color='Prediction',
                    title='Pollution Predictions Over Time',
                    color_discrete_map=APIConfig.LEVEL_COLORS,
                    size=[10] * len(pred_df),
                    hover_data=['Prediction']
                )
                
                st.plotly_chart(fig_pred, width='stretch', key="ts_pred_chart")
    
    with tab2:
        st.markdown("### Spatial Analysis")
        
        # Generate sample spatial data
        np.random.seed(42)
        n_points = 50
        lats = np.random.uniform(25, 45, n_points)
        lons = np.random.uniform(-80, -60, n_points)
        chl_values = np.random.uniform(0.1, 10, n_points)
        
        # Create spatial dataframe
        spatial_df = pd.DataFrame({
            'Latitude': lats,
            'Longitude': lons,
            'CHL': chl_values
        })
        
        # Add simulated pollution levels
        spatial_df['Pollution_Level'] = pd.cut(
            spatial_df['CHL'],
            bins=[0, 1, 3, 100],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Create map
        fig_map = px.scatter_mapbox(
            spatial_df,
            lat='Latitude',
            lon='Longitude',
            color='Pollution_Level',
            size='CHL',
            hover_data=['CHL', 'Pollution_Level'],
            color_discrete_map=APIConfig.LEVEL_COLORS,
            zoom=3,
            height=500,
            title="Pollution Hotspots Map"
        )
        
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        
        st.plotly_chart(fig_map, width='stretch', key="spatial_map")
        
        # Statistics
        col_map1, col_map2, col_map3 = st.columns(3)
        with col_map1:
            st.metric("Total Points", n_points)
        with col_map2:
            high_points = len(spatial_df[spatial_df['Pollution_Level'] == 'HIGH'])
            st.metric("High Risk Areas", high_points)
        with col_map3:
            avg_chl = spatial_df['CHL'].mean()
            st.metric("Average CHL", f"{avg_chl:.2f}")
    
    with tab3:
        st.markdown("### Statistical Analysis")
        
        if st.session_state.predictions_history:
            # Extract data from history
            history_data = []
            for entry in st.session_state.predictions_history:
                if entry.get("result"):
                    history_data.append({
                        'CHL': entry['input'].get('CHL', 0),
                        'Level': entry['result'].get('level', 'UNKNOWN'),
                        'Confidence': entry['result'].get('confidence', 0)
                    })
            
            if history_data:
                stats_df = pd.DataFrame(history_data)
                
                # Display statistics
                st.markdown("#### 📊 Descriptive Statistics")
                st.dataframe(stats_df.describe(), use_container_width=True)
                
                # Visualizations
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    # CHL distribution
                    fig_chl = px.histogram(
                        stats_df,
                        x='CHL',
                        nbins=20,
                        title='CHL Value Distribution',
                        color_discrete_sequence=['#1E88E5']
                    )
                    st.plotly_chart(fig_chl, width='stretch', key="stats_chl_dist")
                
                with col_stat2:
                    # Level distribution
                    level_counts = stats_df['Level'].value_counts()
                    fig_levels = px.bar(
                        x=level_counts.index,
                        y=level_counts.values,
                        title='Prediction Level Distribution',
                        color=level_counts.index,
                        color_discrete_map=APIConfig.LEVEL_COLORS
                    )
                    st.plotly_chart(fig_levels, width='stretch', key="stats_level_dist")
                
                # Correlation analysis
                st.markdown("#### 📈 Correlation Analysis")
                
                # Create correlation matrix (simulated)
                corr_data = pd.DataFrame({
                    'CHL': stats_df['CHL'],
                    'Confidence': stats_df['Confidence'],
                    'Level_Numeric': pd.Categorical(stats_df['Level']).codes
                })
                
                fig_corr = px.imshow(
                    corr_data.corr(),
                    text_auto=True,
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_corr, width='stretch', key="stats_corr_matrix")
            else:
                st.info("No prediction data available for analysis")
        else:
            st.info("Make some predictions first to see statistical analysis")

# Data insights component
def render_data_insights():
    """Render data insights and visualizations"""
    st.markdown('<h1 class="main-header">📈 Data Insights & Visualizations</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Pollution Trends", "🌡️ Environmental Factors", "🎯 Prediction Patterns"])
    
    with tab1:
        st.markdown("### Pollution Level Trends")
        
        # Create sample trend data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Simulated monthly averages
        monthly_avg = {
            'LOW': [45, 42, 38, 35, 30, 25, 20, 22, 28, 35, 40, 43],
            'MEDIUM': [35, 38, 40, 42, 45, 48, 50, 48, 45, 42, 38, 36],
            'HIGH': [20, 20, 22, 23, 25, 27, 30, 30, 27, 23, 22, 21]
        }
        
        # Create trend chart
        fig_trend = go.Figure()
        
        for level in APIConfig.POLLUTION_LEVELS:
            fig_trend.add_trace(go.Scatter(
                x=months,
                y=monthly_avg[level],
                name=level,
                mode='lines+markers',
                line=dict(color=APIConfig.LEVEL_COLORS[level], width=3),
                marker=dict(size=8)
            ))
        
        fig_trend.update_layout(
            title='Monthly Pollution Level Trends',
            xaxis_title='Month',
            yaxis_title='Percentage (%)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, width='stretch', key="insights_trend_chart")
        
        # Insights
        st.markdown("#### 💡 Insights")
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.markdown("""
            **Seasonal Patterns:**
            - HIGH pollution peaks in summer months
            - LOW pollution highest in winter
            - MEDIUM levels relatively stable
            """)
        
        with col_ins2:
            st.markdown("""
            **Recommendations:**
            - Increase monitoring in summer
            - Plan cleanup operations in advance
            - Adjust thresholds seasonally
            """)
    
    with tab2:
        st.markdown("### Environmental Factor Correlations")
        
        # Create correlation data
        factors = ['Temperature', 'Salinity', 'pH', 'Oxygen', 'Nutrients', 'Turbidity']
        correlations = {
            'CHL': [0.65, -0.32, 0.28, -0.45, 0.72, 0.58],
            'PP': [0.58, -0.25, 0.31, -0.38, 0.68, 0.52],
            'KD490': [0.42, -0.18, 0.15, -0.28, 0.45, 0.78]
        }
        
        # Create heatmap
        fig_heat = px.imshow(
            pd.DataFrame(correlations, index=factors),
            text_auto='.2f',
            title='Environmental Factor Correlations',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        st.plotly_chart(fig_heat, width='stretch', key="insights_heatmap")
        
        # Factor importance
        st.markdown("#### 📈 Factor Importance")
        
        importance_data = pd.DataFrame({
            'Factor': factors,
            'Importance': [0.25, 0.15, 0.10, 0.20, 0.18, 0.12]
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_data,
            x='Importance',
            y='Factor',
            orientation='h',
            title='Environmental Factor Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig_importance, width='stretch', key="insights_importance")
    
    with tab3:
        st.markdown("### Prediction Pattern Analysis")
        
        # Create pattern visualization
        patterns = {
            'Clear Pattern': {'LOW': 70, 'MEDIUM': 20, 'HIGH': 10},
            'Moderate Pattern': {'LOW': 30, 'MEDIUM': 50, 'HIGH': 20},
            'Polluted Pattern': {'LOW': 10, 'MEDIUM': 30, 'HIGH': 60},
            'Variable Pattern': {'LOW': 40, 'MEDIUM': 40, 'HIGH': 20}
        }
        
        # Create radar chart
        fig_radar = go.Figure()
        
        for pattern_name, pattern_data in patterns.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=[pattern_data['LOW'], pattern_data['MEDIUM'], pattern_data['HIGH']],
                theta=['LOW', 'MEDIUM', 'HIGH'],
                name=pattern_name,
                fill='toself'
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Prediction Pattern Analysis',
            height=500
        )
        
        st.plotly_chart(fig_radar, width='stretch', key="insights_radar")

# Model information component
def render_model_info():
    """Render model information and diagnostics"""
    st.markdown('<h1 class="main-header">🤖 Model Information & Diagnostics</h1>', unsafe_allow_html=True)
    
    # Get model info from API
    model_info = APIClient.get_model_info()
    
    if model_info and model_info.get("success"):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("### 📊 Model Status")
            
            model_data = model_info.get("model", {})
            metadata = model_info.get("metadata", {})
            
            if model_data.get("loaded"):
                st.success("✅ ML Model Loaded")
                
                # Display model metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Features", model_data.get("features_count", 0))
                with col_met2:
                    accuracy = metadata.get("accuracy", 0)
                    if accuracy:
                        st.metric("Accuracy", f"{float(accuracy):.1%}")
                    else:
                        st.metric("Accuracy", "N/A")
                with col_met3:
                    samples = metadata.get("samples", 0)
                    st.metric("Training Samples", f"{samples:,}")
                
                # Model details
                with st.expander("Model Details", expanded=True):
                    st.json(metadata, expanded=False)
                
                # Features list
                with st.expander("Feature List", expanded=False):
                    features = model_data.get("features", [])
                    if features:
                        for i, feat in enumerate(features[:20], 1):
                            st.write(f"{i}. {feat}")
                        if len(features) > 20:
                            st.write(f"... and {len(features) - 20} more features")
                    else:
                        st.info("No features loaded")
            else:
                st.warning("⚠️ Rule-based Model Active")
                st.info("Using simple threshold-based prediction")
        
        with col_info2:
            st.markdown("### 🔧 Model Diagnostics")
            
            # Test model
            st.markdown("#### Model Testing")
            test_chl = st.number_input("Test CHL Value", value=2.5, key="test_chl")
            
            if st.button("Test Model", icon="🧪"):
                with st.spinner("Testing model..."):
                    result = APIClient.predict_single({"CHL": test_chl})
                
                if result and result.get("success"):
                    prediction = result["prediction"]
                    
                    col_test1, col_test2 = st.columns(2)
                    with col_test1:
                        st.success(f"✅ {prediction['level']}")
                        st.write(f"Confidence: {prediction['confidence']:.1%}")
                    
                    with col_test2:
                        st.write(f"Model: {result.get('model_info', {}).get('model_type', 'N/A')}")
                        st.write(f"Features: {prediction['features_used']}")
                else:
                    st.error("❌ Test failed")
            
            st.markdown("---")
            st.markdown("#### System Information")
            
            # System info
            sys_info = {
                "Python Version": sys.version.split()[0],
                "Platform": sys.platform,
                "API Base URL": APIConfig.BASE_URL,
                "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            for key, value in sys_info.items():
                st.write(f"**{key}:** {value}")
    else:
        st.error("❌ Could not retrieve model information")
        st.info("Make sure the API is running and accessible")

# Prediction history component
def render_prediction_history():
    """Render prediction history interface"""
    st.markdown('<h1 class="main-header">📋 Prediction History</h1>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_history:
        st.info("📭 No prediction history available. Make some predictions first!")
        return
    
    # History statistics
    total = len(st.session_state.predictions_history)
    successful = len([h for h in st.session_state.predictions_history if h.get("result")])
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Predictions", total)
    with col_stat2:
        st.metric("Successful", successful)
    with col_stat3:
        success_rate = successful / total if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1%}")
    with col_stat4:
        # Count HIGH predictions
        high_count = len([
            h for h in st.session_state.predictions_history 
            if h.get("result", {}).get("level") == "HIGH"
        ])
        st.metric("High Risk", high_count)
    
    # Filter and search
    st.markdown("---")
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        filter_level = st.multiselect(
            "Filter by Level:",
            options=APIConfig.POLLUTION_LEVELS,
            default=[]
        )
    
    with col_filter2:
        min_confidence = st.slider(
            "Minimum Confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    with col_filter3:
        items_per_page = st.selectbox(
            "Items per page:",
            options=[10, 25, 50, 100],
            index=0
        )
    
    # Filter history
    filtered_history = st.session_state.predictions_history.copy()
    
    if filter_level:
        filtered_history = [
            h for h in filtered_history 
            if h.get("result", {}).get("level") in filter_level
        ]
    
    if min_confidence > 0:
        filtered_history = [
            h for h in filtered_history 
            if h.get("result", {}).get("confidence", 0) >= min_confidence
        ]
    
    # Pagination
    total_filtered = len(filtered_history)
    page_number = st.number_input(
        "Page:",
        min_value=1,
        max_value=max(1, (total_filtered + items_per_page - 1) // items_per_page),
        value=1,
        step=1
    )
    
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_filtered)
    
    # Display filtered results
    st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {total_filtered} predictions**")
    
    for i in range(start_idx, end_idx):
        entry = filtered_history[i]
        result = entry.get("result", {})
        
        # Create expandable card for each prediction
        with st.expander(f"Prediction {i + 1}: {result.get('level', 'UNKNOWN')} - {result.get('confidence', 0):.1%}", expanded=False):
            col_h1, col_h2, col_h3 = st.columns(3)
            
            with col_h1:
                st.write("**Time:**")
                time_str = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                st.write(time_str)
                
                st.write("**Input Values:**")
                for key, value in entry["input"].items():
                    st.write(f"- {key}: {value}")
            
            with col_h2:
                st.write("**Prediction:**")
                level = result.get("level", "UNKNOWN")
                st.write(f"Level: **{level}**")
                st.write(f"Confidence: **{result.get('confidence', 0):.1%}**")
                st.write(f"CHL Value: **{result.get('chl_value', 'N/A')}**")
                
                st.write("**Probabilities:**")
                probs = result.get("probabilities", {})
                for lvl, prob in probs.items():
                    st.write(f"- {lvl}: {prob:.1%}")
            
            with col_h3:
                st.write("**Model Info:**")
                st.write(f"Features Used: {result.get('features_used', 'N/A')}")
                st.write(f"Model Type: {entry.get('model_info', {}).get('model_type', 'N/A')}")
                
                # Action buttons
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("🔁 Re-run", key=f"rerun_{i}"):
                        # Store for re-run
                        st.session_state.rerun_data = entry["input"]
                        st.rerun()
                
                with col_act2:
                    if st.button("🗑️ Delete", key=f"delete_{i}"):
                        # Remove from history
                        idx = st.session_state.predictions_history.index(entry)
                        st.session_state.predictions_history.pop(idx)
                        st.rerun()
    
    # Export options
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button("📊 Export to CSV", use_container_width=True):
            # Prepare data for export
            export_data = []
            for entry in filtered_history:
                export_data.append({
                    "timestamp": entry["timestamp"],
                    "CHL": entry["input"].get("CHL", ""),
                    "PP": entry["input"].get("PP", ""),
                    "KD490": entry["input"].get("KD490", ""),
                    "level": entry["result"].get("level", ""),
                    "confidence": entry["result"].get("confidence", ""),
                    "features_used": entry["result"].get("features_used", "")
                })
            
            if export_data:
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    with col_exp2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.rerun()
    
    with col_exp3:
        if st.button("📈 Show Charts", use_container_width=True):
            # Show history visualization
            if filtered_history:
                hist_data = []
                for entry in filtered_history:
                    hist_data.append({
                        "time": datetime.fromisoformat(entry["timestamp"]),
                        "level": entry["result"].get("level", "UNKNOWN"),
                        "confidence": entry["result"].get("confidence", 0),
                        "CHL": entry["input"].get("CHL", 0)
                    })
                
                if hist_data:
                    df_hist = pd.DataFrame(hist_data)
                    
                    # Time series of predictions
                    fig_time = px.scatter(
                        df_hist,
                        x="time",
                        y="CHL",
                        color="level",
                        title="Prediction History Timeline",
                        color_discrete_map=APIConfig.LEVEL_COLORS,
                        hover_data=["confidence", "level"]
                    )
                    
                    st.plotly_chart(fig_time, width='stretch', key="history_timeline")

# Settings component
def render_settings():
    """Render settings interface"""
    st.markdown('<h1 class="main-header">⚙️ Settings & Configuration</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["API Settings", "Display Settings", "Advanced"])
    
    with tab1:
        st.markdown("### API Configuration")
        
        current_url = st.text_input(
            "API Base URL:",
            value=APIConfig.BASE_URL,
            help="URL of the Ocean Pollution API server"
        )
        
        # Test connection
        col_test1, col_test2 = st.columns([3, 1])
        with col_test1:
            if st.button("Test API Connection", use_container_width=True):
                try:
                    response = requests.get(f"{current_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ API connection successful")
                        data = response.json()
                        st.json(data, expanded=False)
                    else:
                        st.error(f"❌ API error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        with col_test2:
            if st.button("Reset to Default", use_container_width=True):
                APIConfig.BASE_URL = "http://localhost:8000"
                st.rerun()
        
        # API endpoints
        st.markdown("### Available Endpoints")
        endpoints = [
            ("/", "API Information"),
            ("/docs", "Interactive Documentation"),
            ("/health", "Health Check"),
            ("/model-info", "Model Information"),
            ("/predict", "Single Prediction"),
            ("/predict/batch", "Batch Prediction"),
            ("/project/structure", "Project Structure")
        ]
        
        for endpoint, description in endpoints:
            st.write(f"`{endpoint}` - {description}")
    
    with tab2:
        st.markdown("### Display Settings")
        
        # Theme selection
        theme = st.selectbox(
            "Color Theme:",
            ["Default", "Dark", "Light", "Ocean", "Forest"],
            index=0
        )
        
        # Chart settings
        chart_style = st.selectbox(
            "Chart Style:",
            ["Plotly", "Matplotlib", "Altair"],
            index=0
        )
        
        # Layout settings
        col_layout1, col_layout2 = st.columns(2)
        with col_layout1:
            auto_refresh = st.checkbox("Auto-refresh data", value=False)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh interval (seconds):",
                    min_value=5,
                    max_value=300,
                    value=60,
                    step=5
                )
                st.info(f"Data will refresh every {refresh_interval} seconds")
        
        with col_layout2:
            show_animations = st.checkbox("Show animations", value=True)
            compact_mode = st.checkbox("Compact mode", value=False)
        
        # Apply settings
        if st.button("Apply Display Settings", type="primary"):
            st.success("✅ Display settings applied")
            # Note: In a real app, you would save these to session state or config file
    
    with tab3:
        st.markdown("### Advanced Settings")
        
        # Debug mode
        debug_mode = st.checkbox("Enable debug mode", value=False)
        
        # Log level
        log_level = st.selectbox(
            "Log Level:",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0
        )
        
        # Cache settings
        st.markdown("#### Cache Settings")
        col_cache1, col_cache2 = st.columns(2)
        with col_cache1:
            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Cache cleared")
        
        with col_cache2:
            if st.button("Clear Session State", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Session state cleared")
                st.rerun()
        
        # System information
        st.markdown("#### System Information")
        
        import platform
        sys_info = {
            "Streamlit Version": st.__version__,
            "Python Version": platform.python_version(),
            "Operating System": platform.system(),
            "Processor": platform.processor(),
            "Working Directory": os.getcwd()
        }
        
        for key, value in sys_info.items():
            st.text(f"{key}: {value}")

# Main application
def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content based on selected tab
    current_tab = st.session_state.current_tab
    
    if current_tab == "🏠 Dashboard":
        render_dashboard()
    elif current_tab == "🔬 Real-time Prediction":
        render_realtime_prediction()
    elif current_tab == "📊 Batch Analysis":
        render_batch_analysis()
    elif current_tab == "📈 Data Insights":
        render_data_insights()
    elif current_tab == "🤖 Model Information":
        render_model_info()
    elif current_tab == "📋 Prediction History":
        render_prediction_history()
    elif current_tab == "⚙️ Settings":
        render_settings()
    
    # Footer
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns([1, 2, 1])
    with col_foot2:
        st.markdown(
            '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
            '🌊 Ocean Pollution Dashboard v2.0.0 | '
            'Made with ❤️ for Environmental Protection'
            '</div>',
            unsafe_allow_html=True
        )

# Run the application
if __name__ == "__main__":
    main()
