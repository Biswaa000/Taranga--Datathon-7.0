# Datathon Project

**Author:** Bishal Sharma

## Project Overview

This project participates in a datathon competition focused on predicting death rates across different states, age groups, and time periods. The solution uses machine learning with CatBoost regression to forecast mortality statistics based on various demographic and health-related features.

## Dataset

The project uses three main data files:

- **Train.csv** (11,879 samples): Training data with target variable
- **Test.csv** (5,092 samples): Test data for generating predictions
- **sample_submission.csv**: Expected submission format

### Key Features

- **Demographic Data**: State, sex, age groups (10-year intervals)
- **Temporal Data**: Year, temporal alignment proxies
- **Health Metrics**: Crude rates, death percentages, underlying causes
- **Population Data**: Population counts and adjustments
- **Quality Flags**: Data quality indicators and sensor readings

## Solution Architecture

### Phase 1: Data Cleaning & Preprocessing (`01_basic_draft.ipynb`)

Initial exploratory analysis and basic data cleaning:
- Sex value imputation and encoding
- Missing value handling with appropriate strategies
- Basic data validation

### Phase 2: Optimized Pipeline (`02_optimization.ipynb_`)

Complete production-ready pipeline with:

#### 1. **FastDataProcessor Class**
   - Comprehensive missing value handling
   - Age group ordinal encoding (< 1 year → 85+ years)
   - Feature engineering:
     - State-Year interactions
     - State-Age interactions
     - Death estimates
     - Confidence metrics
     - Demographic indicators (elderly, young)
     - Log-transformed population features
     - Frequency encoding for categorical variables

#### 2. **FastTrainer Class**
   - CatBoost regression with tuned hyperparameters
   - Out-of-fold (OOF) cross-validation evaluation
   - Automatic categorical feature detection
   - Feature importance analysis

### Key Hyperparameters

```python
- Iterations: 1200
- Tree Depth: 8
- Learning Rate: 0.03
- L2 Leaf Regularization: 5
- Min Data in Leaf: 20
- Early Stopping Rounds: 50
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Baseline RMSE** | 14,233.08 |
| **Model RMSE** | 9,401.37 |
| **Improvement** | 33.95% |
| **Training Samples** | 7,459 |
| **Features Used** | 27 |
| **CV Folds** | 5 |

### Top Features by Importance

1. Rate Confidence (15.05)
2. % of Total Deaths (11.48)
3. State-Year Interaction (9.31)
4. Crude Rate Standard Error (6.94)
5. State-Age Interaction (6.91)
6. Death Estimate (5.10)
7. State Target Encoding (5.03)
8. Population (4.53)

### Predictions

- **Total Predictions**: 4,715 rows
- **Prediction Range**: 0 to 42,255
- **Mean Prediction**: 2,574.88
- **Std Dev**: 4,345.71

## Project Structure

```
datathon/
├── README.md                          # This file
├── 01_basic_draft.ipynb               # Initial exploration & cleaning
├── 02_optimization.ipynb_             # Final optimized pipeline
├── Bishal Sharma.docx                 # Project documentation
└── data/
    ├── Train.csv                      # Training dataset
    ├── Test.csv                       # Test dataset
    └── sample_submission.csv          # Submission template
```

## Methodology

### Data Cleaning Strategy

1. **Missing Values**:
   - Sex: Filled from Sex Code, then with mode ('F')
   - Year: Filled from Year Code, then median
   - Population: Filled from adjusted population trend, then median
   - Numeric rates: Filled with column median

2. **Feature Engineering**:
   - Ordinal encoding for age groups (11 categories)
   - Target encoding for State with smoothing (10x factor)
   - Frequency encoding for categorical variables
   - Interaction features between State, Age, and Year

3. **Categorical Handling**:
   - Automatic detection in CatBoost
   - Proper encoding for State, Underlying Cause, Manner of Death

### Model Selection

**CatBoost** was chosen because:
- Native handling of categorical features
- Reduced overfitting through ordered boosting
- Fast training and inference
- Robust performance on tabular data

## Running the Project

### Requirements

```bash
pip install pandas numpy scikit-learn catboost
```

### Execution (Google Colab Environment)

The notebooks are configured to run in Google Colab with:
- Google Drive mounted at `/content/gdrive`
- Data stored in `MyDrive/Datathon/data/`
- Outputs saved to `MyDrive/Datathon/`

### Steps

1. **Data Exploration**: Run `01_basic_draft.ipynb`
   - Understand data structure
   - Identify missing values
   - Analyze distributions

2. **Model Training**: Run `02_optimization.ipynb_`
   - Preprocess data at scale
   - Train and evaluate model
   - Generate submission file

## Key Insights

1. **Feature Importance**: Death rates are primarily driven by confidence in measurements (Rate_Confidence), actual death percentages, and state-specific patterns.

2. **Demographic Patterns**: State and age interactions are significant predictors, suggesting regional and age-based variations.

3. **Data Quality**: Sensor data (temp_sensor_readout, qc_flag_batch_3) have minimal predictive power, indicating clean primary data.

4. **Temporal Effects**: Year-based features show consistent importance across model iterations.

## Improvements & Next Steps

Potential enhancements:
- Ensemble methods combining multiple models
- Advanced feature engineering with domain expertise
- Hyperparameter optimization (Bayesian search)
- Handling of outliers with robust scaling
- Time-series specific features (trends, seasonality)
- External data integration (healthcare datasets)

## Contact

For questions or details about this project, refer to Bishal Sharma's documentation or the code comments within the notebooks.

---

*Last Updated: January 2026*
