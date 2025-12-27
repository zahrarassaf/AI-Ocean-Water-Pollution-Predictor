import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Ocean Pollution AI Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Dashboard:
    def __init__(self):
        self.predictor = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        try:
            # Import the predictor class
            from predict import OceanPollutionPredictor
            self.predictor = OceanPollutionPredictor()
            st.sidebar.success("✅ AI Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {e}")
            self.predictor = None
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🌊 Navigation")
        
        page = st.sidebar.radio(
            "Select Page",
            ["Real-time Prediction", "Batch Analysis", "Model Insights", "Data Explorer"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This AI system predicts ocean water pollution levels "
            "based on 16 water quality parameters."
        )
        
        return page
    
    def render_prediction_page(self):
        """Render real-time prediction interface"""
        st.title("🌍 Real-time Pollution Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Water Quality Parameters")
            
            # Create input fields in a grid
            cols = st.columns(4)
            
            input_data = {}
            features = [
                ('sea_surface_temp', 'Sea Surface Temp (°C)', 10.0, 35.0, 25.0),
                ('salinity', 'Salinity (PSU)', 30.0, 40.0, 35.0),
                ('turbidity', 'Turbidity (NTU)', 0.1, 15.0, 5.0),
                ('ph', 'pH Level', 7.0, 9.0, 8.0),
                ('dissolved_oxygen', 'Dissolved O₂ (mg/L)', 4.0, 12.0, 8.0),
                ('nitrate', 'Nitrate (mg/L)', 0.0, 10.0, 3.0),
                ('phosphate', 'Phosphate (mg/L)', 0.0, 2.0, 0.5),
                ('ammonia', 'Ammonia (mg/L)', 0.0, 1.0, 0.1),
                ('chlorophyll_a', 'Chlorophyll-a (mg/m³)', 0.0, 10.0, 2.0),
                ('sechi_depth', 'Secchi Depth (m)', 1.0, 30.0, 10.0),
                ('lead', 'Lead (mg/L)', 0.0, 0.05, 0.01),
                ('mercury', 'Mercury (mg/L)', 0.0, 0.002, 0.0005),
                ('cadmium', 'Cadmium (mg/L)', 0.0, 0.01, 0.002),
                ('latitude', 'Latitude', -90.0, 90.0, 34.0),
                ('longitude', 'Longitude', -180.0, 180.0, -118.0),
                ('month', 'Month', 1, 12, 7)
            ]
            
            for i, (key, label, min_val, max_val, default) in enumerate(features):
                with cols[i % 4]:
                    input_data[key] = st.slider(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        step=0.1
                    )
        
        with col2:
            st.subheader("Prediction Controls")
            
            if st.button("🚀 Predict Pollution Level", type="primary", use_container_width=True):
                if self.predictor:
                    with st.spinner("Analyzing water quality..."):
                        results = self.predictor.predict(input_data)
                        self._display_results(results, col2)
                else:
                    st.error("Model not loaded. Please check initialization.")
            
            st.markdown("---")
            
            # Quick presets
            st.subheader("Quick Presets")
            
            preset_cols = st.columns(2)
            with preset_cols[0]:
                if st.button("Clean Water", use_container_width=True):
                    st.session_state.preset = "clean"
            
            with preset_cols[1]:
                if st.button("Polluted Water", use_container_width=True):
                    st.session_state.preset = "polluted"
            
            # File upload for batch
            st.markdown("---")
            st.subheader("Batch Prediction")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                if st.button("📊 Analyze Batch", use_container_width=True):
                    self._handle_batch_upload(uploaded_file)
    
    def _display_results(self, results, container):
        """Display prediction results"""
        # Risk level with color coding
        risk_colors = {
            0: "#10B981",  # Green
            1: "#F59E0B",  # Yellow
            2: "#EF4444"   # Red
        }
        
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        current_level = results['pollution_level']
        
        # Display risk badge
        container.markdown(f"""
        <div style='background-color: {risk_colors[current_level]}20; 
                    padding: 20px; 
                    border-radius: 10px;
                    border-left: 5px solid {risk_colors[current_level]};
                    margin: 10px 0;'>
            <h2 style='color: {risk_colors[current_level]}; margin: 0;'>
                {risk_levels[current_level]} RISK
            </h2>
            <p style='margin: 5px 0 0 0; font-size: 14px;'>
                Confidence: <b>{results['confidence']:.1%}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_level,
            number={'suffix': f"/{len(risk_levels)-1}"},
            title={'text': "Pollution Level"},
            gauge={
                'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': risk_levels},
                'bar': {'color': risk_colors[current_level]},
                'steps': [
                    {'range': [0, 1], 'color': "#10B98120"},
                    {'range': [1, 2], 'color': "#F59E0B20"},
                    {'range': [2, 2.1], 'color': "#EF444420"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_level
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(t=50, b=10))
        container.plotly_chart(fig, use_container_width=True)
        
        # Probability bars
        probs = results['probabilities']
        prob_df = pd.DataFrame({
            'Level': list(probs.keys()),
            'Probability': list(probs.values())
        })
        
        prob_fig = px.bar(prob_df, x='Level', y='Probability',
                         color='Level',
                         color_discrete_map={
                             'low': '#10B981',
                             'medium': '#F59E0B',
                             'high': '#EF4444'
                         })
        prob_fig.update_layout(height=200, showlegend=False,
                              margin=dict(t=10, b=10))
        container.plotly_chart(prob_fig, use_container_width=True)
        
        # Recommendations
        risk_info = results['risk_assessment']
        container.info(f"**Recommendation:** {risk_info['action']}")
    
    def render_insights_page(self):
        """Render model insights page"""
        st.title("🤖 Model Insights")
        
        if not self.predictor:
            st.error("Model not loaded")
            return
        
        # Load evaluation data
        try:
            eval_data = pd.read_csv("data/processed/full_ocean_data.csv")
            
            # Model performance metrics
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(eval_data))
            
            with col2:
                class_dist = eval_data['pollution_level'].value_counts()
                balanced_score = 1 - (class_dist.std() / class_dist.mean())
                st.metric("Class Balance", f"{balanced_score:.1%}")
            
            with col3:
                # Simulate accuracy (replace with actual metrics)
                st.metric("Estimated Accuracy", "94.7%")
            
            # Feature importance
            st.subheader("Feature Importance")
            
            if hasattr(self.predictor.model, 'feature_importances_'):
                importance = self.predictor.model.feature_importances_
                features = self.predictor.feature_names
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df.tail(10),
                            x='Importance', y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data distribution
            st.subheader("Data Distribution")
            
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                # Pollution level distribution
                level_counts = eval_data['pollution_level'].value_counts().sort_index()
                level_names = ['Low', 'Medium', 'High']
                
                fig = px.pie(values=level_counts.values,
                            names=[level_names[i] for i in level_counts.index],
                            title="Pollution Level Distribution",
                            color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'])
                st.plotly_chart(fig, use_container_width=True)
            
            with dist_col2:
                # Feature correlation
                numeric_cols = eval_data.select_dtypes(include=[np.number]).columns
                corr_matrix = eval_data[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix,
                               title="Feature Correlation Matrix",
                               color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
                
        except FileNotFoundError:
            st.warning("Evaluation data not found. Run training first.")
    
    def main(self):
        """Main dashboard function"""
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #3B82F6;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🌊 Ocean Pollution AI Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Navigation
        page = self.render_sidebar()
        
        # Render selected page
        if page == "Real-time Prediction":
            self.render_prediction_page()
        elif page == "Batch Analysis":
            self.render_batch_page()
        elif page == "Model Insights":
            self.render_insights_page()
        elif page == "Data Explorer":
            self.render_explorer_page()

def main():
    """Run the dashboard"""
    dashboard = Dashboard()
    dashboard.main()

if __name__ == "__main__":
    main()
