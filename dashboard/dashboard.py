import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

st.set_page_config(
    page_title="Ocean Pollution Predictor",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_dirs = glob.glob('../models/final_model_*')
    if not model_dirs:
        return None
    
    model_dirs.sort(key=os.path.getmtime, reverse=True)
    model_path = os.path.join(model_dirs[0], 'final_model.joblib')
    
    return joblib.load(model_path)

def main():
    st.title("🌊 AI Ocean Water Pollution Predictor")
    st.markdown("---")
    
    model_data = load_model()
    
    if model_data is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    metrics = model_data['metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{metrics.get('test_r2', metrics.get('r2', 0)):.4f}")
    
    with col2:
        st.metric("RMSE", f"{metrics.get('test_rmse', metrics.get('rmse', 0)):.4f}")
    
    with col3:
        st.metric("MAE", f"{metrics.get('test_mae', metrics.get('mae', 0)):.4f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Predict", "📈 Analysis", "ℹ️ Model Info"])
    
    with tab1:
        st.header("Make Prediction")
        
        features = {}
        cols = st.columns(4)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 4]:
                features[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=0.01,
                    key=f"input_{feature}"
                )
        
        if st.button("🚀 Predict", type="primary"):
            input_array = np.array([features[f] for f in feature_names]).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            
            st.success(f"**Prediction:** {prediction[0]:.6f}")
            
            st.subheader("Feature Importance for this Prediction")
            
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
            ax.set_title("Top 10 Feature Importances")
            st.pyplot(fig)
    
    with tab2:
        st.header("Model Analysis")
        
        if 'feature_importance' in model_data:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Features")
                st.dataframe(importance_df.head(10))
            
            with col2:
                st.subheader("Feature Importance Chart")
                fig, ax = plt.subplots(figsize=(8, 6))
                importance_df.head(15).plot.barh(x='Feature', y='Importance', ax=ax)
                ax.invert_yaxis()
                st.pyplot(fig)
        
        if 'predictions.csv' in os.listdir(model_dirs[0]):
            predictions_path = os.path.join(model_dirs[0], 'predictions.csv')
            predictions_df = pd.read_csv(predictions_path)
            
            st.subheader("Predictions Distribution")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.scatter(predictions_df['actual'][:1000], 
                       predictions_df['predicted'][:1000], 
                       alpha=0.5, s=10)
            ax1.plot([predictions_df['actual'].min(), predictions_df['actual'].max()],
                    [predictions_df['actual'].min(), predictions_df['actual'].max()],
                    'r--')
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title('Actual vs Predicted')
            
            ax2.hist(predictions_df['error'], bins=50, alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Error')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Error Distribution')
            
            st.pyplot(fig)
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Training Date:** {model_data.get('training_date', 'N/A')}")
            st.write(f"**Model Type:** {type(model).__name__}")
            st.write(f"**Number of Trees:** {model.n_estimators}")
            st.write(f"**Max Depth:** {model.max_depth}")
            st.write(f"**Number of Features:** {len(feature_names)}")
        
        with col2:
            st.subheader("Performance Metrics")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    st.write(f"**{key}:** {value:.4f}")
                else:
                    st.write(f"**{key}:** {value}")
        
        st.subheader("All Features")
        st.write(", ".join(feature_names))

if __name__ == "__main__":
    main()
