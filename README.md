# Titanic Survival Prediction

A comprehensive machine learning solution for predicting passenger survival on the Titanic using various classification algorithms.

## Features

- **Data Preprocessing**: Handles missing values with appropriate imputation strategies
- **Feature Engineering**: Creates meaningful features like FamilySize, IsAlone, and AgeGroup
- **Categorical Encoding**: Proper encoding of categorical variables
- **Multiple Models**: Compares Logistic Regression, Random Forest, and SVM
- **Cross-Validation**: Robust model evaluation using 5-fold cross-validation
- **Visualization**: Comprehensive data exploration and result visualization
- **Prediction Pipeline**: Ready-to-use prediction function for new passengers

## Installation

1. Clone or download this project
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from titanic_prediction import TitanicSurvivalPredictor

# Create predictor instance
predictor = TitanicSurvivalPredictor()

# Run complete pipeline
predictor.run_complete_pipeline()
```

### Step-by-Step Usage

```python
# Load data (will create sample dataset if train.csv not found)
predictor.load_data()

# Explore the dataset
predictor.explore_data()

# Preprocess data with imputation and encoding
predictor.preprocess_data()

# Train multiple models
predictor.train_models()

# Evaluate the best model
predictor.evaluate_best_model()
```

### Making Predictions

```python
# Example passenger data
passenger_data = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 50,
    'Embarked': 'S'
}

# Make prediction
result = predictor.predict_survival(passenger_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Survival Probability: {result['survival_probability']:.4f}")
```

## Data Features

The model uses the following features:

- **Pclass**: Passenger class (1, 2, 3)
- **Sex**: Gender (male, female)
- **Age**: Passenger age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (S, C, Q)

### Engineered Features

- **FamilySize**: Total family members aboard
- **IsAlone**: Whether passenger is traveling alone
- **AgeGroup**: Categorized age groups (Child, Teen, Adult, Senior)

## Models Compared

1. **Logistic Regression**: Baseline linear classifier
2. **Random Forest**: Ensemble decision tree model
3. **Support Vector Machine**: Non-linear classifier

## Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **Cross-Validation Score**: 5-fold CV for robust evaluation
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation of predictions
- **Feature Importance**: For Random Forest model

## Output Files

The pipeline generates several visualization files:

- `titanic_exploration.png`: Data exploration plots
- `confusion_matrix.png`: Confusion matrix heatmap
- `feature_importance.png`: Feature importance plot (Random Forest)

## Sample Dataset

If you don't have the Titanic dataset, the code will automatically generate a realistic sample dataset with:
- 891 passengers (same as original dataset)
- Realistic distributions for all features
- Appropriate missing values to demonstrate imputation

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Project Structure

```
titanic_survival_prediction/
├── titanic_prediction.py    # Main prediction class
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── titanic_exploration.png # Generated exploration plots
├── confusion_matrix.png    # Generated confusion matrix
└── feature_importance.png  # Generated feature importance plot
```

## Example Output

```
=== Titanic Survival Prediction Pipeline ===
Sample dataset created with shape: (891, 12)

=== Dataset Info ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          791 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        594 non-null    object 
 11  Embarked     841 non-null    object 
dtypes: float64(2), int64(5), object(5)

Training Logistic Regression...
CV Score: 0.7856 (+/- 0.0824)
Test Accuracy: 0.8101

Training Random Forest...
CV Score: 0.8154 (+/- 0.0692)
Test Accuracy: 0.8326

Training SVM...
CV Score: 0.8032 (+/- 0.0748)
Test Accuracy: 0.8212

Best model: Random Forest with accuracy 0.8326

Example Prediction: Survived
Survival Probability: 0.7854
```

## Contributing

Feel free to modify and improve the code! Some potential enhancements:
- Add more models (XGBoost, LightGBM, Neural Networks)
- Implement hyperparameter tuning
- Add more sophisticated feature engineering
- Include model ensemble methods
- Add web interface for predictions
