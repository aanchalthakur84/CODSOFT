import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class TitanicSurvivalPredictor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        
    def load_data(self, train_path='c:\\Users\\VICTUS\\ARPAN DOC\\Titanic-Dataset.csv', test_path='test.csv'):
        """Load Titanic dataset"""
        try:
            self.data = pd.read_csv(train_path)
            print(f"Dataset loaded successfully from {train_path} with shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Dataset not found at {train_path}. Creating sample dataset for demonstration...")
            self._create_sample_dataset()
            return True
    
    def _create_sample_dataset(self):
        """Create a sample Titanic dataset for demonstration"""
        np.random.seed(42)
        n_samples = 891
        
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.binomial(1, 0.38, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
            'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 15, n_samples),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]),
            'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
            'Fare': np.random.exponential(30, n_samples) + 10,
            'Cabin': [f'Cabin_{i%100}' if i%3 != 0 else np.nan for i in range(1, n_samples + 1)],
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        self.data = pd.DataFrame(data)
        
        # Add some missing values to make it realistic
        self.data.loc[np.random.choice(self.data.index, 100), 'Age'] = np.nan
        self.data.loc[np.random.choice(self.data.index, 50), 'Embarked'] = np.nan
        
        print(f"Sample dataset created with shape: {self.data.shape}")
    
    def explore_data(self):
        """Explore the dataset"""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("\n=== Dataset Info ===")
        print(self.data.info())
        
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
        
        print("\n=== Basic Statistics ===")
        print(self.data.describe())
        
        print("\n=== Survival Rate ===")
        print(self.data['Survived'].value_counts(normalize=True))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Survival by Sex
        sns.countplot(data=self.data, x='Sex', hue='Survived', ax=axes[0, 0])
        axes[0, 0].set_title('Survival by Sex')
        
        # Survival by Pclass
        sns.countplot(data=self.data, x='Pclass', hue='Survived', ax=axes[0, 1])
        axes[0, 1].set_title('Survival by Passenger Class')
        
        # Age distribution
        sns.histplot(data=self.data, x='Age', hue='Survived', multiple='stack', ax=axes[1, 0])
        axes[1, 0].set_title('Age Distribution by Survival')
        
        # Fare distribution
        sns.histplot(data=self.data, x='Fare', hue='Survived', multiple='stack', ax=axes[1, 1])
        axes[1, 1].set_title('Fare Distribution by Survival')
        
        plt.tight_layout()
        plt.savefig('titanic_exploration.png')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the data with imputation and encoding"""
        if self.data is None:
            print("Please load data first!")
            return
        
        # Select relevant features
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = self.data[features].copy()
        y = self.data['Survived']
        
        # Handle missing values
        # Numerical imputation
        numerical_features = ['Age', 'Fare']
        numerical_imputer = SimpleImputer(strategy='median')
        X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
        
        # Categorical imputation
        categorical_features = ['Sex', 'Embarked']
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # Encode categorical variables
        label_encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
            label_encoders[feature] = le
        
        # Feature engineering
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
        X['AgeGroup'] = LabelEncoder().fit_transform(X['AgeGroup'])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # Store preprocessing components
        self.preprocessor = {
            'numerical_imputer': numerical_imputer,
            'categorical_imputer': categorical_imputer,
            'label_encoders': label_encoders,
            'scaler': scaler
        }
        
        print(f"Data preprocessed successfully!")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple classification models"""
        if self.X_train is None:
            print("Please preprocess data first!")
            return
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            # Train on full training set
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        self.best_model = results[best_model_name]['model']
        print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['test_accuracy']:.4f}")
    
    def evaluate_best_model(self):
        """Evaluate the best performing model"""
        if self.best_model is None:
            print("Please train models first!")
            return
        
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_accuracy'])
        results = self.models[best_model_name]
        
        print(f"\n=== Evaluation for {best_model_name} ===")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"CV Score: {results['cv_score']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, results['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, results['predictions'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Survived', 'Survived'],
                   yticklabels=['Not Survived', 'Survived'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Feature importance (for Random Forest)
        if best_model_name == 'Random Forest':
            feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'AgeGroup']
            importances = self.best_model.feature_importances_
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances, y=feature_names)
            plt.title('Feature Importance - Random Forest')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
    
    def predict_survival(self, passenger_data):
        """Make predictions for new passengers"""
        if self.best_model is None:
            print("Please train models first!")
            return
        
        # Convert to DataFrame if needed
        if isinstance(passenger_data, dict):
            passenger_data = pd.DataFrame([passenger_data])
        
        # Preprocess the input data
        X = passenger_data.copy()
        
        # Handle missing values
        numerical_features = ['Age', 'Fare']
        X[numerical_features] = self.preprocessor['numerical_imputer'].transform(X[numerical_features])
        
        categorical_features = ['Sex', 'Embarked']
        X[categorical_features] = self.preprocessor['categorical_imputer'].transform(X[categorical_features])
        
        # Encode categorical variables
        for feature in categorical_features:
            X[feature] = self.preprocessor['label_encoders'][feature].transform(X[feature])
        
        # Feature engineering
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
        X['AgeGroup'] = LabelEncoder().fit_transform(X['AgeGroup'])
        
        # Scale features
        X_scaled = self.preprocessor['scaler'].transform(X)
        
        # Make prediction
        prediction = self.best_model.predict(X_scaled)
        probability = self.best_model.predict_proba(X_scaled)[:, 1]
        
        return {
            'prediction': prediction[0],
            'survival_probability': probability[0],
            'prediction_label': 'Survived' if prediction[0] == 1 else 'Not Survived'
        }
    
    def run_complete_pipeline(self):
        """Run the complete prediction pipeline"""
        print("=== Titanic Survival Prediction Pipeline ===")
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Evaluate best model
        self.evaluate_best_model()
        
        print("\n=== Pipeline completed successfully! ===")

# Example usage
if __name__ == "__main__":
    predictor = TitanicSurvivalPredictor()
    predictor.run_complete_pipeline()
    
    # Example prediction
    example_passenger = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 50,
        'Embarked': 'S'
    }
    
    result = predictor.predict_survival(example_passenger)
    print(f"\nExample Prediction: {result['prediction_label']}")
    print(f"Survival Probability: {result['survival_probability']:.4f}")
