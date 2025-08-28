"""
Simplified EV Analysis without matplotlib dependencies
Focuses on core machine learning functionality
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the EV dataset"""
    print("Loading Washington State EV Registration Dataset...")
    df = pd.read_csv('Electric_Vehicle_Population_Data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Handle missing values
    print("Handling missing values...")
    
    # For categorical features, fill with 'Unknown'
    categorical_features = ['County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Utility']
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('Unknown')
    
    # For numerical features, fill with median
    numerical_features = ['Model Year', 'Postal Code', 'Base MSRP', 'Legislative District', 'Electric Range']
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median())
    
    # Create binary CAFV eligibility target
    print("Creating binary CAFV eligibility target...")
    df['CAFV_Eligible'] = (df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] == 
                          'Clean Alternative Fuel Vehicle Eligible').astype(int)
    
    # Remove rows where CAFV eligibility is unknown for classification task
    cafv_known = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] != 'Eligibility unknown as battery range has not been researched'
    df_cafv = df[cafv_known].copy()
    
    # Remove rows where electric range is 0 or missing for regression task
    range_valid = (df['Electric Range'] > 0) & (df['Electric Range'].notna())
    df_range = df[range_valid].copy()
    
    print(f"Dataset for CAFV classification: {df_cafv.shape}")
    print(f"Dataset for range prediction: {df_range.shape}")
    
    return df_cafv, df_range

def create_features(df):
    """Create engineered features"""
    print("Creating engineered features...")
    
    # Extract location features from Vehicle Location
    if 'Vehicle Location' in df.columns:
        # Parse POINT coordinates
        df['Longitude'] = df['Vehicle Location'].str.extract(r'POINT \(([-\d.]+)')
        df['Latitude'] = df['Vehicle Location'].str.extract(r'([-\d.]+)\)')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    
    # Create age feature
    current_year = 2024
    df['Vehicle_Age'] = current_year - df['Model Year']
    
    # Create price category
    df['Price_Category'] = pd.cut(df['Base MSRP'], 
                                 bins=[0, 30000, 50000, 80000, float('inf')], 
                                 labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
    
    return df

def prepare_ml_data(df_cafv, df_range):
    """Prepare data for machine learning models"""
    print("Preparing ML data...")
    
    # Features for both tasks
    feature_columns = [
        'Model Year', 'Make', 'Model', 'Electric Vehicle Type', 
        'County', 'Electric Utility', 'Vehicle_Age', 'Price_Category'
    ]
    
    # Add location features if available
    if 'Longitude' in df_cafv.columns and 'Latitude' in df_cafv.columns:
        feature_columns.extend(['Longitude', 'Latitude'])
    
    # Prepare CAFV classification data
    X_cafv = df_cafv[feature_columns].copy()
    y_cafv = df_cafv['CAFV_Eligible']
    
    # Prepare range prediction data
    X_range = df_range[feature_columns].copy()
    y_range = df_range['Electric Range']
    
    # Encode categorical variables
    categorical_columns = X_cafv.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns to encode: {list(categorical_columns)}")
    
    for col in categorical_columns:
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        X_cafv[col] = le.fit_transform(X_cafv[col].astype(str))
        X_range[col] = le.transform(X_range[col].astype(str))
    
    # Ensure all columns are numeric
    print(f"Final data types: {X_cafv.dtypes}")
    
    # Handle any remaining NaN values
    print(f"NaN values in X_cafv: {X_cafv.isnull().sum().sum()}")
    print(f"NaN values in X_range: {X_range.isnull().sum().sum()}")
    
    # Fill any remaining NaN values with 0
    X_cafv = X_cafv.fillna(0)
    X_range = X_range.fillna(0)
    
    print(f"CAFV features shape: {X_cafv.shape}")
    print(f"Range features shape: {X_range.shape}")
    print(f"CAFV target distribution: {y_cafv.value_counts().to_dict()}")
    
    return X_cafv, y_cafv, X_range, y_range

def train_cafv_classifier(X, y):
    """Train CAFV classification models"""
    print("\n" + "="*60)
    print("TASK 1: CAFV ELIGIBILITY CLASSIFICATION")
    print("="*60)
    
    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weight_dict),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight_dict),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        
        results[name] = {
            'model': model,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"F1 Score: {f1:.4f}")
        print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Select best model
    best_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_name]['model']
    
    print(f"\nBest Model: {best_name}")
    print(f"Best F1 Score: {results[best_name]['f1_score']:.4f}")
    
    # Detailed evaluation of best model
    print(f"\nDetailed Evaluation of {best_name}:")
    print("Classification Report:")
    print(classification_report(y_test, results[best_name]['predictions']))
    
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, results[best_name]['predictions'])
    print(cm)
    
    return results, best_model, scaler

def train_range_regressor(X, y):
    """Train electric range prediction models"""
    print("\n" + "="*60)
    print("TASK 2: ELECTRIC RANGE PREDICTION")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate RMSE and R²
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
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
    
    # Select best model
    best_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = results[best_name]['model']
    
    print(f"\nBest Model: {best_name}")
    print(f"Best RMSE: {results[best_name]['rmse']:.2f}")
    
    return results, best_model, scaler

def main():
    """Main function to run the analysis"""
    print("="*80)
    print("WASHINGTON STATE EV REGISTRATION DATASET ANALYSIS")
    print("="*80)
    
    # Load and preprocess data
    df_cafv, df_range = load_and_preprocess_data()
    
    # Create features
    df_cafv = create_features(df_cafv)
    df_range = create_features(df_range)
    
    # Prepare ML data
    X_cafv, y_cafv, X_range, y_range = prepare_ml_data(df_cafv, df_range)
    
    # Train CAFV classification models
    cafv_results, cafv_best_model, cafv_scaler = train_cafv_classifier(X_cafv, y_cafv)
    
    # Train range prediction models
    range_results, range_best_model, range_scaler = train_range_regressor(X_range, y_range)
    
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
    
    # Save results
    print("\nSaving results...")
    
    # Save processed data
    X_cafv.to_csv('X_cafv.csv', index=False)
    y_cafv.to_csv('y_cafv.csv', index=False)
    X_range.to_csv('X_range.csv', index=False)
    y_range.to_csv('y_range.csv', index=False)
    
    # Save models
    import joblib
    joblib.dump({
        'model': cafv_best_model,
        'scaler': cafv_scaler,
        'feature_names': X_cafv.columns.tolist()
    }, 'cafv_best_model.pkl')
    
    joblib.dump({
        'model': range_best_model,
        'scaler': range_scaler,
        'feature_names': X_range.columns.tolist()
    }, 'range_best_model.pkl')
    
    print("✓ Analysis completed successfully!")
    print("✓ Models saved as .pkl files")
    print("✓ Processed data saved as .csv files")
    
    return cafv_results, range_results

if __name__ == "__main__":
    cafv_results, range_results = main()
