import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class CAFVClassifier:
    """CAFV Eligibility Classification Model"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X, y):
        """Prepare data for training"""
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, class_weight_dict
    
    def train_models(self, X, y):
        """Train all models and select the best one"""
        print("Training CAFV Classification Models...")
        
        X_train, X_test, y_train, y_test, class_weight_dict = self.prepare_data(X, y)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Set class weights for models that support it
            if hasattr(model, 'class_weight'):
                if name == 'Random Forest':
                    model.class_weight = class_weight_dict
                elif name == 'Logistic Regression':
                    model.class_weight = class_weight_dict
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate F1 score
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            
            results[name] = {
                'model': model,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"F1 Score: {f1:.4f}")
            print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model based on F1 score
        best_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_name]['model']
        
        print(f"\nBest Model: {best_name}")
        print(f"Best F1 Score: {results[best_name]['f1_score']:.4f}")
        
        # Detailed evaluation of best model
        self.evaluate_model(results[best_name], y_test, best_name)
        
        return results
    
    def evaluate_model(self, model_result, y_test, model_name):
        """Evaluate the best model in detail"""
        print(f"\nDetailed Evaluation of {model_name}:")
        print("="*50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, model_result['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, model_result['predictions'])
        print(f"Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Eligible', 'Eligible'],
                   yticklabels=['Not Eligible', 'Eligible'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'cafv_confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()
        
        # Feature importance (if available)
        if hasattr(model_result['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model_result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'cafv_feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300)
            plt.show()
    
    def save_model(self, filename='cafv_best_model.pkl'):
        """Save the best model"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, filename)
            print(f"Model saved as {filename}")

class RangeRegressor:
    """Electric Range Prediction Model"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X, y):
        """Prepare data for training"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X, y):
        """Train all models and select the best one"""
        print("Training Electric Range Prediction Models...")
        
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate RMSE and R²
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2_score': r2,
                'cv_rmse': cv_rmse,
                'cv_rmse_std': np.sqrt(cv_scores.var()),
                'predictions': y_pred
            }
            
            print(f"RMSE: {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            print(f"CV RMSE: {cv_rmse:.2f} (+/- {np.sqrt(cv_scores.var()) * 2:.2f})")
        
        # Select best model based on RMSE
        best_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        self.best_model = results[best_name]['model']
        
        print(f"\nBest Model: {best_name}")
        print(f"Best RMSE: {results[best_name]['rmse']:.2f}")
        
        # Detailed evaluation of best model
        self.evaluate_model(results[best_name], y_test, best_name)
        
        return results
    
    def evaluate_model(self, model_result, y_test, model_name):
        """Evaluate the best model in detail"""
        print(f"\nDetailed Evaluation of {model_name}:")
        print("="*50)
        
        # Prediction vs Actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, model_result['predictions'], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Electric Range')
        plt.ylabel('Predicted Electric Range')
        plt.title(f'Prediction vs Actual - {model_name}')
        plt.text(0.05, 0.95, f'R² = {model_result["r2_score"]:.4f}\nRMSE = {model_result["rmse"]:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig(f'range_prediction_plot_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()
        
        # Residuals plot
        residuals = y_test - model_result['predictions']
        plt.figure(figsize=(10, 6))
        plt.scatter(model_result['predictions'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Electric Range')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(f'range_residuals_plot_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()
        
        # Feature importance (if available)
        if hasattr(model_result['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model_result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'range_feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300)
            plt.show()
    
    def save_model(self, filename='range_best_model.pkl'):
        """Save the best model"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, filename)
            print(f"Model saved as {filename}")

def main():
    """Main function to train both models"""
    print("Loading processed data...")
    
    # Load processed data
    X_cafv = pd.read_csv('X_cafv.csv')
    y_cafv = pd.read_csv('y_cafv.csv').squeeze()
    X_range = pd.read_csv('X_range.csv')
    y_range = pd.read_csv('y_range.csv').squeeze()
    
    print(f"CAFV data shape: {X_cafv.shape}")
    print(f"Range data shape: {X_range.shape}")
    
    # Train CAFV classification model
    print("\n" + "="*60)
    print("TASK 1: CAFV ELIGIBILITY CLASSIFICATION")
    print("="*60)
    
    cafv_classifier = CAFVClassifier()
    cafv_results = cafv_classifier.train_models(X_cafv, y_cafv)
    cafv_classifier.save_model()
    
    # Train Range prediction model
    print("\n" + "="*60)
    print("TASK 2: ELECTRIC RANGE PREDICTION")
    print("="*60)
    
    range_regressor = RangeRegressor()
    range_results = range_regressor.train_models(X_range, y_range)
    range_regressor.save_model()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nCAFV Classification Results:")
    for name, result in cafv_results.items():
        print(f"{name}: F1 Score = {result['f1_score']:.4f}")
    
    print("\nRange Prediction Results:")
    for name, result in range_results.items():
        print(f"{name}: RMSE = {result['rmse']:.2f}, R² = {result['r2_score']:.4f}")
    
    return cafv_classifier, range_regressor

if __name__ == "__main__":
    cafv_classifier, range_regressor = main()
