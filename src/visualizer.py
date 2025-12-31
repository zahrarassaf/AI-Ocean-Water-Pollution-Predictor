import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PollutionVisualizer:
    def __init__(self, style: str = 'dark'):
        self.style = style
        self.set_style()
    
    def set_style(self):
        """Set visualization style."""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'LOW': '#2ecc71',     # Green
                'MEDIUM': '#f39c12',  # Orange
                'HIGH': '#e74c3c',    # Red
                'CRITICAL': '#8b0000' # Dark Red
            }
        else:
            self.colors = {
                'LOW': '#27ae60',
                'MEDIUM': '#f1c40f',
                'HIGH': '#e74c3c',
                'CRITICAL': '#c0392b'
            }
    
    def plot_distribution(self, data: pd.DataFrame, 
                         target_col: str = 'CHL',
                         save_path: Optional[str] = None):
        """Plot distribution of pollution levels."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(data[target_col], bins=50, 
                       color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'Distribution of {target_col}')
        axes[0, 0].set_xlabel(target_col)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(data[target_col].mean(), color='red', 
                          linestyle='--', label=f'Mean: {data[target_col].mean():.2f}')
        
        # Box plot
        axes[0, 1].boxplot(data[target_col].dropna(), vert=False)
        axes[0, 1].set_title(f'Box Plot of {target_col}')
        axes[0, 1].set_xlabel(target_col)
        
        # QQ plot
        from scipy import stats
        stats.probplot(data[target_col].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Violin plot (if categories exist)
        if 'pollution_level' in data.columns:
            sns.violinplot(x='pollution_level', y=target_col, 
                         data=data, ax=axes[1, 1], palette=self.colors)
            axes[1, 1].set_title('Pollution Levels Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                              save_path: Optional[str] = None):
        """Plot correlation matrix of features."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) > 15:
            # Select top correlated features
            corr_matrix = numeric_data.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > 0.95)]
            numeric_data = numeric_data.drop(columns=to_drop)
        
        plt.figure(figsize=(12, 10))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, annot=True, fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_interactive_map(self, data: pd.DataFrame, 
                             lat_col: str = 'lat',
                             lon_col: str = 'lon',
                             value_col: str = 'CHL',
                             save_path: Optional[str] = None):
        """Create interactive Folium map."""
        # Create base map
        m = folium.Map(location=[data[lat_col].mean(), 
                               data[lon_col].mean()],
                      zoom_start=5,
                      tiles='CartoDB dark_matter')
        
        # Add heatmap
        heat_data = [[row[lat_col], row[lon_col], row[value_col]] 
                    for _, row in data.iterrows()]
        
        plugins.HeatMap(heat_data, 
                       min_opacity=0.2,
                       max_opacity=0.8,
                       radius=15,
                       blur=10,
                       max_zoom=1).add_to(m)
        
        # Add markers for extreme values
        extreme_threshold = data[value_col].quantile(0.95)
        extreme_data = data[data[value_col] > extreme_threshold]
        
        for _, row in extreme_data.iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=8,
                popup=f"{value_col}: {row[value_col]:.2f}",
                color='red',
                fill=True,
                fill_color='red'
            ).add_to(m)
        
        if save_path:
            m.save(save_path)
        
        return m
    
    def plot_time_series_forecast(self, forecast_df: pd.DataFrame,
                                actual_df: Optional[pd.DataFrame] = None,
                                save_path: Optional[str] = None):
        """Plot time series forecast with Plotly."""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Pollution Level Forecast', 
                                         'Confidence Levels'),
                           vertical_spacing=0.15)
        
        # Pollution levels
        levels = forecast_df['predicted_level'].unique()
        
        for level in levels:
            level_data = forecast_df[forecast_df['predicted_level'] == level]
            fig.add_trace(
                go.Scatter(x=level_data['date'], y=[level]*len(level_data),
                          mode='markers',
                          name=level,
                          marker=dict(color=self.colors.get(level, '#000000'),
                                     size=10),
                          showlegend=True),
                row=1, col=1
            )
        
        # Confidence levels
        fig.add_trace(
            go.Scatter(x=forecast_df['date'], y=forecast_df['confidence'],
                      mode='lines+markers',
                      name='Confidence',
                      line=dict(color='cyan', width=2),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        # Add actual data if provided
        if actual_df is not None:
            fig.add_trace(
                go.Scatter(x=actual_df['date'], y=actual_df['value'],
                          mode='lines',
                          name='Actual',
                          line=dict(color='yellow', width=2, dash='dash')),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Ocean Pollution 7-Day Forecast',
            template='plotly_dark' if self.style == 'dark' else 'plotly_white',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_yaxes(title_text='Pollution Level', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Confidence', row=2, col=1, range=[0, 1])
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard_visualizations(self, data: pd.DataFrame,
                                      predictions: Dict,
                                      output_dir: str = "reports/"):
        """Create comprehensive dashboard visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        # 1. Distribution plot
        dist_path = output_path / "distribution.png"
        self.plot_distribution(data, save_path=str(dist_path))
        visualizations['distribution'] = str(dist_path)
        
        # 2. Correlation matrix
        corr_path = output_path / "correlation.png"
        self.plot_correlation_matrix(data, save_path=str(corr_path))
        visualizations['correlation'] = str(corr_path)
        
        # 3. Prediction results
        if predictions and 'predictions' in predictions:
            pred_df = pd.DataFrame(predictions['predictions'])
            
            # Bar chart of predictions
            plt.figure(figsize=(10, 6))
            pred_counts = pred_df['prediction'].value_counts()
            colors = [self.colors.get(level, '#000000') 
                     for level in pred_counts.index]
            
            plt.bar(pred_counts.index, pred_counts.values, color=colors)
            plt.title('Prediction Distribution', fontsize=14)
            plt.xlabel('Pollution Level')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            pred_path = output_path / "predictions.png"
            plt.savefig(pred_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['predictions'] = str(pred_path)
        
        # 4. Confidence distribution
        if predictions and 'predictions' in predictions:
            confidences = [p['confidence'] for p in predictions['predictions']]
            
            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=20, color='skyblue', 
                    edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(confidences), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
            plt.title('Confidence Distribution', fontsize=14)
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            
            conf_path = output_path / "confidence.png"
            plt.savefig(conf_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['confidence'] = str(conf_path)
        
        logger.info(f"Visualizations saved to {output_path}")
        return visualizations
