# Washington State EV Registration Dataset - Machine Learning Analysis

This project analyzes Washington State's Electric Vehicle (EV) registration dataset to build machine learning models for two key prediction tasks.

## Objectives

### Task 1: CAFV Eligibility Classification
- **Goal**: Predict whether a vehicle is Clean Alternative Fuel Vehicle (CAFV) eligible
- **Type**: Binary Classification
- **Metric**: F1-score (handles class imbalance)
- **Features**: Model year, make, model, vehicle type, electric range, location, utility

### Task 2: Electric Range Prediction
- **Goal**: Predict the vehicle's electric range in miles
- **Type**: Regression
- **Metric**: RMSE (Root Mean Squared Error)
- **Features**: Model year, make, model, vehicle type, location, utility

## Dataset

The dataset contains 257,635 electric vehicle registrations from Washington State with the following key information:
- Vehicle details (make, model, year, type)
- CAFV eligibility status
- Electric range
- Location data (county, city, GPS coordinates)
- Utility information

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the dataset file: `Electric_Vehicle_Population_Data.csv`

## Usage

### Run Complete Analysis
```bash
python main.py
```

This will:
1. Perform data exploration and preprocessing
2. Train multiple ML models for both tasks
3. Evaluate model performance
4. Generate visualizations
5. Save trained models
6. Create a comprehensive report

### Individual Components

#### Data Exploration
```bash
python data_exploration.py
```

#### Model Training
```bash
python ml_models.py
```

## Project Structure

```
├── main.py                          # Main analysis script
├── data_exploration.py              # Data exploration and preprocessing
├── ml_models.py                     # Machine learning models
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── Electric_Vehicle_Population_Data.csv  # Dataset (you need to provide this)
├── analysis_report.md               # Generated analysis report
├── *.png                           # Generated visualizations
├── *.pkl                           # Saved trained models
└── *.csv                           # Processed datasets
```

## Models Used

### Classification Models (CAFV Eligibility)
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Machine (SVM)

### Regression Models (Electric Range)
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Support Vector Regression (SVR)

## Key Features

- **Class Imbalance Handling**: Uses class weights and F1-score for imbalanced classification
- **Feature Engineering**: Extracts location features from GPS coordinates
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, feature importance
- **Visualization**: Generates plots for model performance and data distributions
- **Model Persistence**: Saves trained models for future predictions

## Results

The analysis generates:
- Model performance metrics (F1-score for classification, RMSE for regression)
- Feature importance rankings
- Confusion matrices and classification reports
- Prediction vs actual plots
- Residual analysis
- Comprehensive markdown report

## Technical Details

- **Python Version**: 3.7+
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Data Split**: 80/20 train/test split
- **Feature Scaling**: StandardScaler applied to all features
- **Missing Value Handling**: Appropriate imputation strategies

## License

This project is for educational and research purposes. Please ensure you have the right to use the Washington State EV registration dataset.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this analysis.
