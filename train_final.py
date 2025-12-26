import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_curve, auc, roc_auc_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("OCEAN WATER POLLUTION PREDICTION MODEL")
print("=" * 60)

class OceanPollutionModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        
    def load_data(self):
        """بارگذاری داده‌ها"""
        print("\n📊 Loading data...")
        
        try:
            self.X_train = pd.read_csv("data/processed/X_train.csv")
            self.X_test = pd.read_csv("data/processed/X_test.csv")
            self.y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
            self.y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
            
            print(f"✅ Data loaded successfully!")
            print(f"   Training set: {self.X_train.shape}")
            print(f"   Test set: {self.X_test.shape}")
            print(f"   Features: {list(self.X_train.columns)}")
            
            # نمایش توزیع کلاس‌ها
            print(f"\n📈 Class distribution:")
            train_counts = self.y_train.value_counts().sort_index()
            test_counts = self.y_test.value_counts().sort_index()
            
            for label in [0, 1, 2]:
                print(f"   Class {label} (Low/Med/High): Train={train_counts.get(label, 0)}, Test={test_counts.get(label, 0)}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please run process_data.py first")
            return False
    
    def preprocess_data(self):
        """پیش‌پردازش داده‌ها"""
        print("\n🔧 Preprocessing data...")
        
        # ذخیره نام ستون‌ها
        self.feature_names = self.X_train.columns.tolist()
        
        # نرمال‌سازی داده‌ها
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Data preprocessing completed!")
        return True
    
    def train_models(self):
        """آموزش مدل‌های مختلف"""
        print("\n🤖 Training multiple models...")
        
        # تعریف مدل‌ها
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n   Training {name}...")
            
            # آموزش مدل
            model.fit(self.X_train_scaled, self.y_train)
            
            # پیش‌بینی
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # محاسبه معیارها
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"     Accuracy: {accuracy:.4f}")
            print(f"     CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # انتخاب بهترین مدل
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        # ذخیره اهمیت ویژگی‌ها برای Random Forest
        if best_model_name == 'Random Forest':
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.results = results
        return results
    
    def evaluate_models(self):
        """ارزیابی جامع مدل‌ها"""
        print("\n📈 Comprehensive model evaluation...")
        
        # ایجاد نمودار
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. مقایسه دقت مدل‌ها
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, accuracies, width, label='Test Accuracy', color='skyblue')
        axes[0].bar(x + width/2, cv_means, width, label='CV Mean', color='lightcoral')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. اهمیت ویژگی‌ها (اگر Random Forest بهترین بود)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1].barh(range(len(top_features)), top_features['importance'])
            axes[1].set_yticks(range(len(top_features)))
            axes[1].set_yticklabels(top_features['feature'])
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Top 10 Important Features')
            axes[1].invert_yaxis()
        
        # 3. ماتریس درهم‌ریختگی برای بهترین مدل
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        y_pred_best = self.results[best_model_name]['y_pred']
        
        cm = confusion_matrix(self.y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'], ax=axes[2])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title(f'Confusion Matrix - {best_model_name}')
        
        # 4. گزارش طبقه‌بندی
        axes[3].axis('off')
        report = classification_report(self.y_test, y_pred_best, 
                                      target_names=['Low', 'Medium', 'High'])
        axes[3].text(0, 0.5, report, fontfamily='monospace', fontsize=10, 
                    verticalalignment='center')
        axes[3].set_title('Classification Report')
        
        # 5. ROC Curve (برای کلاس‌های مختلف)
        if self.results[best_model_name]['y_pred_proba'] is not None:
            y_proba = self.results[best_model_name]['y_pred_proba']
            
            # One-vs-Rest ROC curves
            for i in range(3):
                fpr, tpr, _ = roc_curve((self.y_test == i).astype(int), y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                axes[4].plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            axes[4].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[4].set_xlabel('False Positive Rate')
            axes[4].set_ylabel('True Positive Rate')
            axes[4].set_title('ROC Curves (One-vs-Rest)')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        
        # 6. توزیع خطاها
        error_indices = np.where(y_pred_best != self.y_test)[0]
        if len(error_indices) > 0:
            error_counts = self.y_test.iloc[error_indices].value_counts()
            axes[5].pie(error_counts.values, labels=error_counts.index, 
                       autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
            axes[5].set_title('Error Distribution by Class')
        
        plt.tight_layout()
        plt.savefig('models/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # نمایش نتایج عددی
        print("\n" + "=" * 60)
        print("FINAL MODEL PERFORMANCE")
        print("=" * 60)
        
        for name in model_names:
            result = self.results[name]
            print(f"\n{name}:")
            print(f"  Test Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Accuracy:   {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
        
        # نمایش اهمیت ویژگی‌ها
        if self.feature_importance is not None:
            print(f"\n📊 TOP 5 IMPORTANT FEATURES:")
            for idx, row in self.feature_importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self):
        """ذخیره مدل‌ها"""
        print("\n💾 Saving models...")
        
        # ایجاد پوشه models
        os.makedirs("models", exist_ok=True)
        
        # ذخیره بهترین مدل
        joblib.dump(self.best_model, "models/best_ocean_pollution_model.pkl")
        
        # ذخیره همه مدل‌ها
        for name, result in self.results.items():
            joblib.dump(result['model'], f"models/{name.lower().replace(' ', '_')}.pkl")
        
        # ذخیره scaler
        joblib.dump(self.scaler, "models/scaler.pkl")
        
        # ذخیره نام ویژگی‌ها
        with open("models/feature_names.txt", "w") as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        print("✅ Models saved in 'models' directory:")
        print("   - best_ocean_pollution_model.pkl")
        print("   - All individual models")
        print("   - scaler.pkl")
        print("   - feature_names.txt")
    
    def create_prediction_example(self):
        """ایجاد مثال پیش‌بینی"""
        print("\n🔮 Creating prediction example...")
        
        # انتخاب یک نمونه تصادفی از تست
        idx = np.random.randint(0, len(self.X_test))
        sample = self.X_test.iloc[idx].values.reshape(1, -1)
        actual = self.y_test.iloc[idx]
        
        # مقیاس‌سازی
        sample_scaled = self.scaler.transform(sample)
        
        # پیش‌بینی
        prediction = self.best_model.predict(sample_scaled)[0]
        probabilities = self.best_model.predict_proba(sample_scaled)[0]
        
        # نمایش نتایج
        pollution_levels = {0: "Low", 1: "Medium", 2: "High"}
        
        print(f"\nSample prediction:")
        print(f"  Actual pollution level: {pollution_levels[actual]} ({actual})")
        print(f"  Predicted level: {pollution_levels[prediction]} ({prediction})")
        
        print(f"\nPrediction probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  {pollution_levels[i]}: {prob:.2%}")
        
        # نمایش مقادیر ویژگی‌ها
        print(f"\nFeature values:")
        top_features = self.feature_importance.head(5)['feature'].tolist() if self.feature_importance is not None else self.feature_names[:5]
        
        for feature in top_features:
            if feature in self.X_test.columns:
                value = self.X_test.iloc[idx][feature]
                print(f"  {feature}: {value:.4f}")
        
        return sample, actual, prediction, probabilities

def main():
    """تابع اصلی"""
    
    # ایجاد مدل
    ocean_model = OceanPollutionModel()
    
    # بارگذاری داده
    if not ocean_model.load_data():
        return
    
    # پیش‌پردازش
    ocean_model.preprocess_data()
    
    # آموزش مدل‌ها
    ocean_model.train_models()
    
    # ارزیابی
    ocean_model.evaluate_models()
    
    # ذخیره مدل‌ها
    ocean_model.save_models()
    
    # ایجاد مثال
    ocean_model.create_prediction_example()
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check 'models' folder for saved models")
    print("2. Run 'predict.py' for new predictions")
    print("3. Check 'model_evaluation.png' for visualizations")

if __name__ == "__main__":
    main()
