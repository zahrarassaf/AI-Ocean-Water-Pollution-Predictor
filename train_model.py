"""
Train and save the pollution prediction model
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_training_data():
    """Generate sample training data"""
    np.random.seed(42)
    
    n_samples = 1000
    data = []
    
    for _ in range(n_samples):
        # Random values for features
        chl = np.random.uniform(0.1, 20.0)
        prod = np.random.uniform(50, 1500)
        trans = np.random.uniform(1, 35)
        
        # Determine pollution level based on chlorophyll
        if chl <= 1.0:
            pollution_level = 0  # LOW
        elif chl <= 5.0:
            pollution_level = 1  # MEDIUM
        else:
            pollution_level = 2  # HIGH
        
        data.append([chl, prod, trans, pollution_level])
    
    df = pd.DataFrame(data, columns=['CHL_diffuse_at', 'PP_primary_pr', 'TRANS_secchi_dep', 'pollution_level'])
    return df

def train_and_save_model():
    print("Training pollution prediction model...")
    
    # Generate training data
    df = generate_training_data()
    
    # Prepare features and target
    X = df[['CHL_diffuse_at', 'PP_primary_pr', 'TRANS_secchi_dep']].values
    y = df['pollution_level'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/pollution_model.pkl')
    print("Model saved to 'models/pollution_model.pkl'")
    
    # Also save as default location
    joblib.dump(model, 'pollution_model.pkl')
    print("Model also saved to 'pollution_model.pkl'")
    
    return model

if __name__ == "__main__":
    train_and_save_model()
