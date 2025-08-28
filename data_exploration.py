import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform initial exploration of the EV dataset"""
    print("Loading Washington State EV Registration Dataset...")
    df = pd.read_csv('Electric_Vehicle_Population_Data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def analyze_target_variables(df):
    """Analyze the target variables for both tasks"""
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    # CAFV Eligibility Analysis
    print("\n1. CAFV Eligibility Distribution:")
    cafv_counts = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
    print(cafv_counts)
    print(f"Class imbalance ratio: {cafv_counts.max() / cafv_counts.min():.2f}:1")
    
    # Electric Range Analysis
    print("\n2. Electric Range Statistics:")
    range_stats = df['Electric Range'].describe()
    print(range_stats)
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # CAFV Eligibility distribution
    cafv_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('CAFV Eligibility Distribution')
    axes[0,0].set_xlabel('CAFV Eligibility')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Electric Range histogram
    df['Electric Range'].hist(bins=50, ax=axes[0,1], color='lightgreen', alpha=0.7)
    axes[0,1].set_title('Electric Range Distribution')
    axes[0,1].set_xlabel('Electric Range (miles)')
    axes[0,1].set_ylabel('Frequency')
    
    # Electric Range by Vehicle Type
    df.boxplot(column='Electric Range', by='Electric Vehicle Type', ax=axes[1,0])
    axes[1,0].set_title('Electric Range by Vehicle Type')
    axes[1,0].set_xlabel('Vehicle Type')
    axes[1,0].set_ylabel('Electric Range (miles)')
    
    # Electric Range by Make (top 10)
    top_makes = df['Make'].value_counts().head(10).index
    df_top_makes = df[df['Make'].isin(top_makes)]
    df_top_makes.boxplot(column='Electric Range', by='Make', ax=axes[1,1])
    axes[1,1].set_title('Electric Range by Top 10 Makes')
    axes[1,1].set_xlabel('Make')
    axes[1,1].set_ylabel('Electric Range (miles)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_features(df):
    """Analyze feature distributions and relationships"""
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    # Categorical features
    categorical_features = ['County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Utility']
    
    print("\nCategorical Feature Distributions:")
    for feature in categorical_features:
        if feature in df.columns:
            unique_count = df[feature].nunique()
            print(f"{feature}: {unique_count} unique values")
            if unique_count <= 20:
                print(f"  Values: {df[feature].value_counts().head(10).to_dict()}")
    
    # Numerical features
    numerical_features = ['Model Year', 'Postal Code', 'Base MSRP', 'Legislative District']
    print(f"\nNumerical Feature Statistics:")
    print(df[numerical_features].describe())
    
    # Missing values analysis
    print(f"\nMissing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    df_processed = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    
    # For categorical features, fill with 'Unknown'
    categorical_features = ['County', 'City', 'State', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Utility']
    for feature in categorical_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna('Unknown')
    
    # For numerical features, fill with median
    numerical_features = ['Model Year', 'Postal Code', 'Base MSRP', 'Legislative District', 'Electric Range']
    for feature in numerical_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())
    
    # Create binary CAFV eligibility target
    print("Creating binary CAFV eligibility target...")
    df_processed['CAFV_Eligible'] = (df_processed['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] == 
                                   'Clean Alternative Fuel Vehicle Eligible').astype(int)
    
    # Remove rows where CAFV eligibility is unknown for classification task
    cafv_known = df_processed['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] != 'Eligibility unknown as battery range has not been researched'
    df_cafv = df_processed[cafv_known].copy()
    
    # Remove rows where electric range is 0 or missing for regression task
    range_valid = (df_processed['Electric Range'] > 0) & (df_processed['Electric Range'].notna())
    df_range = df_processed[range_valid].copy()
    
    print(f"Dataset for CAFV classification: {df_cafv.shape}")
    print(f"Dataset for range prediction: {df_range.shape}")
    
    return df_cafv, df_range

def create_features(df):
    """Create engineered features for both tasks"""
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
    
    # Create range category for analysis
    df['Range_Category'] = pd.cut(df['Electric Range'], 
                                 bins=[0, 50, 150, 250, float('inf')], 
                                 labels=['Short', 'Medium', 'Long', 'Very Long'])
    
    return df

def prepare_ml_data(df_cafv, df_range):
    """Prepare data for machine learning models"""
    print("\n" + "="*50)
    print("PREPARING ML DATA")
    print("="*50)
    
    # Features for both tasks
    feature_columns = [
        'Model Year', 'Make', 'Model', 'Electric Vehicle Type', 
        'County', 'Electric Utility', 'Vehicle_Age', 'Price_Category'
    ]
    
    # Add location features if available
    if 'Longitude' in df_cafv.columns and 'Latitude' in df_cafv.columns:
        feature_columns.extend(['Longitude', 'Latitude'])
    
    # Prepare CAFV classification data
    print("Preparing CAFV classification data...")
    X_cafv = df_cafv[feature_columns].copy()
    y_cafv = df_cafv['CAFV_Eligible']
    
    # Prepare range prediction data
    print("Preparing range prediction data...")
    X_range = df_range[feature_columns].copy()
    y_range = df_range['Electric Range']
    
    # Encode categorical variables
    categorical_columns = X_cafv.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_cafv[col] = le.fit_transform(X_cafv[col].astype(str))
        X_range[col] = le.transform(X_range[col].astype(str))
    
    print(f"CAFV features shape: {X_cafv.shape}")
    print(f"Range features shape: {X_range.shape}")
    print(f"CAFV target distribution: {y_cafv.value_counts().to_dict()}")
    
    return X_cafv, y_cafv, X_range, y_range

def main():
    """Main function to run data exploration and preprocessing"""
    # Load data
    df = load_and_explore_data()
    
    # Analyze target variables
    analyze_target_variables(df)
    
    # Analyze features
    analyze_features(df)
    
    # Preprocess data
    df_cafv, df_range = preprocess_data(df)
    
    # Create features
    df_cafv = create_features(df_cafv)
    df_range = create_features(df_range)
    
    # Prepare ML data
    X_cafv, y_cafv, X_range, y_range = prepare_ml_data(df_cafv, df_range)
    
    # Save processed data
    print("\nSaving processed data...")
    X_cafv.to_csv('X_cafv.csv', index=False)
    y_cafv.to_csv('y_cafv.csv', index=False)
    X_range.to_csv('X_range.csv', index=False)
    y_range.to_csv('y_range.csv', index=False)
    
    print("Data exploration and preprocessing completed!")
    return X_cafv, y_cafv, X_range, y_range

if __name__ == "__main__":
    X_cafv, y_cafv, X_range, y_range = main()
