import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OceanPollutionPredictor:
    """
    AI-powered ocean water pollution prediction system.
    Uses trained machine learning models to predict pollution levels.
    """
    
    def __init__(self):
        """Initialize the predictor with trained models"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_components()
    
    def _load_components(self):
        """Load trained model and preprocessing components"""
        try:
            print("🔄 Loading AI model components...")
            
            self.model = joblib.load("models/best_ocean_pollution_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            
            with open("models/feature_names.txt", "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            model_type = type(self.model).__name__
            print(f"✅ Model loaded successfully: {model_type}")
            print(f"📊 Features: {len(self.feature_names)} parameters")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please train the model first using train_model.py")
            raise
    
    def predict(self, input_data):
        """
        Predict pollution level from input data.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            dict: Prediction results with probabilities
        """
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Ensure correct feature order
        df = df[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        # Prepare results
        results = {
            'pollution_level': int(prediction),
            'confidence': float(np.max(probabilities)),
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'risk_assessment': self._assess_risk(prediction, probabilities),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _assess_risk(self, prediction, probabilities):
        """Assess environmental risk based on prediction"""
        risk_levels = {
            0: {
                'level': 'LOW',
                'color': '#10B981',
                'action': 'Normal monitoring',
                'impact': 'Minimal environmental impact'
            },
            1: {
                'level': 'MEDIUM',
                'color': '#F59E0B',
                'action': 'Increased monitoring',
                'impact': 'Moderate ecological stress'
            },
            2: {
                'level': 'HIGH',
                'color': '#EF4444',
                'action': 'Immediate investigation',
                'impact': 'Significant environmental risk'
            }
        }
        
        return risk_levels[prediction]
    
    def predict_batch(self, data_file):
        """Predict pollution levels for multiple samples"""
        print(f"\n📁 Processing batch predictions from: {data_file}")
        
        # Load data
        data = pd.read_csv(data_file)
        
        # Validate columns
        missing_cols = set(self.feature_names) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Make predictions
        scaled_data = self.scaler.transform(data[self.feature_names])
        predictions = self.model.predict(scaled_data)
        probabilities = self.model.predict_proba(scaled_data)
        
        # Create results DataFrame
        results = data.copy()
        results['predicted_level'] = predictions
        results['confidence'] = np.max(probabilities, axis=1)
        
        for i in range(3):
            results[f'prob_level_{i}'] = probabilities[:, i]
        
        # Save results
        output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(output_file, index=False)
        
        print(f"✅ Batch predictions saved to: {output_file}")
        print(f"📈 Distribution: {pd.Series(predictions).value_counts().to_dict()}")
        
        return results
    
    def visualize_prediction(self, input_data, results):
        """Create visualization for prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Ocean Pollution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pollution Level Gauge
        ax1 = axes[0, 0]
        self._create_gauge_chart(ax1, results)
        
        # 2. Probability Distribution
        ax2 = axes[0, 1]
        self._create_probability_chart(ax2, results)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        self._create_feature_chart(ax3, input_data)
        
        # 4. Risk Assessment
        ax4 = axes[1, 1]
        self._create_risk_chart(ax4, results)
        
        plt.tight_layout()
        plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_gauge_chart(self, ax, results):
        """Create gauge chart for pollution level"""
        levels = ['LOW', 'MEDIUM', 'HIGH']
        colors = ['#10B981', '#F59E0B', '#EF4444']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
        
        # Fill current level
        current_level = results['pollution_level']
        theta_fill = np.linspace(0, (current_level + 1) * np.pi/3, 50)
        ax.fill_between(np.cos(theta_fill), 0, np.sin(theta_fill), 
                       alpha=0.3, color=colors[current_level])
        
        # Add pointer
        pointer_angle = (current_level + 0.5) * np.pi/3
        ax.arrow(0, 0, 0.7*np.cos(pointer_angle), 0.7*np.sin(pointer_angle),
                head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Pollution Level: {levels[current_level]}', 
                    fontsize=14, fontweight='bold', color=colors[current_level])
    
    def _create_probability_chart(self, ax, results):
        """Create probability distribution chart"""
        probs = results['probabilities']
        levels = list(probs.keys())
        values = list(probs.values())
        colors = ['#10B981', '#F59E0B', '#EF4444']
        
        bars = ax.bar(levels, values, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Prediction Confidence', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    def _create_feature_chart(self, ax, input_data):
        """Display top features influencing prediction"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            features = self.feature_names
            
            # Get top 5 features
            idx = np.argsort(importance)[-5:]
            top_features = [features[i] for i in idx]
            top_importance = importance[idx]
            
            bars = ax.barh(top_features, top_importance, color='steelblue', alpha=0.7)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Top Influencing Factors', fontsize=14)
            ax.invert_yaxis()
            
            # Add value labels
            for bar, val in zip(bars, top_importance):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10)
    
    def _create_risk_chart(self, ax, results):
        """Display risk assessment information"""
        risk = results['risk_assessment']
        
        # Create colored risk box
        ax.add_patch(plt.Rectangle((0.1, 0.6), 0.8, 0.3, 
                                  color=risk['color'], alpha=0.2, ec=risk['color'], lw=2))
        
        ax.text(0.5, 0.8, risk['level'], fontsize=24, fontweight='bold',
               ha='center', va='center', color=risk['color'])
        
        ax.text(0.5, 0.65, risk['action'], fontsize=12,
               ha='center', va='center', style='italic')
        
        ax.text(0.5, 0.4, 'Recommended Action:', fontsize=11,
               ha='center', va='center', fontweight='bold')
        
        ax.text(0.5, 0.3, risk['action'], fontsize=10,
               ha='center', va='center', wrap=True)
        
        ax.text(0.5, 0.15, f"Impact: {risk['impact']}", fontsize=10,
               ha='center', va='center', style='italic')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Risk Assessment', fontsize=14)

def generate_sample_data():
    """Generate realistic ocean water quality sample data"""
    return {
        'sea_surface_temp': 28.5,
        'salinity': 35.8,
        'turbidity': 7.2,
        'ph': 8.1,
        'dissolved_oxygen': 6.8,
        'nitrate': 5.3,
        'phosphate': 1.1,
        'ammonia': 0.3,
        'chlorophyll_a': 7.8,
        'sechi_depth': 6.5,
        'lead': 0.035,
        'mercury': 0.0018,
        'cadmium': 0.007,
        'latitude': 34.0522,
        'longitude': -118.2437,
        'month': 8
    }

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🌊 AI OCEAN WATER POLLUTION PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = OceanPollutionPredictor()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return
    
    # Example 1: Single prediction with sample data
    print("\n📋 Example 1: Single Prediction")
    print("-" * 40)
    
    sample_data = generate_sample_data()
    print("\nSample Water Quality Parameters:")
    for key, value in sample_data.items():
        print(f"  {key:20}: {value:8.4f}")
    
    # Make prediction
    results = predictor.predict(sample_data)
    
    print(f"\n🔍 Prediction Results:")
    print(f"  Pollution Level: {results['pollution_level']}")
    print(f"  Confidence: {results['confidence']:.2%}")
    print(f"  Risk Level: {results['risk_assessment']['level']}")
    
    print("\n📊 Probability Distribution:")
    for level, prob in results['probabilities'].items():
        print(f"  {level.upper():8}: {prob:.2%}")
    
    # Create visualization
    print("\n🖼️  Generating visualization...")
    predictor.visualize_prediction(sample_data, results)
    
    # Example 2: Batch prediction
    print("\n📋 Example 2: Batch Prediction")
    print("-" * 40)
    
    # Create sample batch data
    np.random.seed(42)
    n_samples = 50
    batch_data = pd.DataFrame({
        feature: np.random.rand(n_samples) for feature in predictor.feature_names
    })
    
    # Scale to realistic ranges
    batch_data['sea_surface_temp'] = batch_data['sea_surface_temp'] * 25 + 10
    batch_data['salinity'] = batch_data['salinity'] * 8 + 30
    batch_data['chlorophyll_a'] = batch_data['chlorophyll_a'] * 10
    
    # Save sample batch
    batch_data.to_csv('sample_batch_data.csv', index=False)
    
    # Run batch prediction
    try:
        batch_results = predictor.predict_batch('sample_batch_data.csv')
        
        # Summary statistics
        print(f"\n📈 Batch Prediction Summary:")
        print(f"  Total samples: {len(batch_results)}")
        level_counts = batch_results['predicted_level'].value_counts()
        for level, count in level_counts.items():
            percentage = count / len(batch_results) * 100
            risk_level = ['LOW', 'MEDIUM', 'HIGH'][level]
            print(f"  {risk_level:7}: {count:3d} samples ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Batch prediction skipped: {e}")
    
    print("\n" + "="*60)
    print("✅ PREDICTION SYSTEM READY")
    print("="*60)
    
    # Usage instructions
    print("\n🚀 How to use this system:")
    print("""
    1. For single predictions:
       predictor = OceanPollutionPredictor()
       result = predictor.predict(your_data_dict)
    
    2. For batch predictions:
       results = predictor.predict_batch('your_data.csv')
    
    3. Data format requirements:
       - Use the 16 features listed in feature_names.txt
       - Values should be in proper units (temperature in °C, etc.)
       - Missing values should be handled before prediction
    """)

if __name__ == "__main__":
    main()
