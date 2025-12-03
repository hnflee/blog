# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
# import seaborn as sns  # Commented out as it's not available
from typing import Dict, Tuple, Optional
import warnings
import argparse
import os
import pickle
import json
from scipy.stats import randint, uniform
warnings.filterwarnings('ignore')


class XGBoostTrainer:
    """XGBoost classification model trainer with validation and hyperparameter tuning"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.feature_names = None
        self.best_params = None
        self.cv_results = None
        self.label_encoder = None
        self.feature_encoders = {}  # Store encoders for categorical features

    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and testing data"""
        print("Preparing data...")

        # Read data
        df = pd.read_csv(data_path)
        print(f"Original dataset shape: {df.shape}")

        # Handle missing values
        df_clean = df.dropna()
        print(f"After removing missing values: {df_clean.shape}")

        # Separate features and target
        target_column = df_clean.columns[-1]
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]

        # Filter out classes with fewer than 2 samples
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index
        if len(rare_classes) > 0:
            print(f"Removing {len(rare_classes)} classes with fewer than 2 samples")
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]
        
        # Analyze class distribution
        class_counts = y.value_counts()
        print(f"\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(y)*100:.2f}%)")
        
        # Store class weights for imbalanced data handling
        self.class_weights = {}
        total_samples = len(y)
        n_classes = len(class_counts)
        for cls, count in class_counts.items():
            # Inverse frequency weighting
            self.class_weights[cls] = total_samples / (n_classes * count)

        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Handle categorical features (object type columns)
        print("Processing categorical features...")
        for col in X.columns:
            if X[col].dtype == 'object':
                # Handle boolean-like columns (if_first_carrier, if_last_carrier)
                if col in ['if_first_carrier', 'if_last_carrier']:
                    print(f"  Converting {col} from boolean/string to numeric")
                    X[col] = X[col].map({'true': 1, 'false': 0, True: 1, False: 0, 'True': 1, 'False': 0})
                    # Fill any remaining NaN values with 0
                    X[col] = X[col].fillna(0).astype(int)
                else:
                    # Use LabelEncoder for other categorical columns
                    print(f"  Encoding {col} using LabelEncoder")
                    if col not in self.feature_encoders:
                        self.feature_encoders[col] = LabelEncoder()
                    X[col] = self.feature_encoders[col].fit_transform(X[col].astype(str))
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"  Warning: {col} is still object type, converting to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Convert to numeric types
        X = X.astype(float)

        self.feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        # Convert to numpy arrays (XGBoost accepts both DataFrame and array, but we use array for consistency)
        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")

        return X_train, X_test, y_train, y_test

    def train_basic_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         use_early_stopping: bool = True, 
                         use_class_weights: bool = True) -> xgb.XGBClassifier:
        """Train basic XGBoost classification model with optimizations"""
        print("\nTraining basic XGBoost model...")

        # Calculate sample weights if class weights are available
        sample_weight = None
        if use_class_weights and hasattr(self, 'class_weights') and self.class_weights:
            print("Using class weights for imbalanced data...")
            # Map class labels to weights
            weight_map = {self.label_encoder.transform([cls])[0]: weight 
                         for cls, weight in self.class_weights.items()}
            sample_weight = np.array([weight_map[label] for label in y_train])
            print(f"Sample weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")

        # Improved default parameters
        self.model = xgb.XGBClassifier(
            n_estimators=300,  # Increased from 100
            max_depth=6,
            learning_rate=0.05,  # Reduced from 0.1 for better convergence
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,  # Added for regularization
            gamma=0.1,  # Added for regularization
            reg_alpha=0.1,  # Added L1 regularization
            reg_lambda=1.0,  # Added L2 regularization
            random_state=self.random_state,
            n_jobs=-1,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=len(self.label_encoder.classes_),
            tree_method='hist',  # Faster training
            verbosity=1
        )

        # Use early stopping if validation set is available
        if use_early_stopping:
            # Split training data for validation
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, 
                stratify=y_train
            )
            
            # Split sample weights accordingly
            train_sample_weight = None
            if sample_weight is not None:
                # Use the same split indices by splitting indices first
                indices = np.arange(len(X_train))
                train_indices, val_indices = train_test_split(
                    indices, test_size=0.2, random_state=self.random_state, 
                    stratify=y_train
                )
                train_sample_weight = sample_weight[train_indices]
            
            # Use early_stopping_rounds parameter (compatible with all XGBoost versions)
            self.model.fit(
                X_train_fit, y_train_fit,
                sample_weight=train_sample_weight,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=True
            )
            
            # Get best iteration and score
            if hasattr(self.model, 'best_iteration'):
                print(f"Best iteration: {self.model.best_iteration}")
            if hasattr(self.model, 'best_score'):
                print(f"Best score: {self.model.best_score:.4f}")
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        print("Basic model training completed.")
        return self.model

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             method: str = 'randomized', n_iter: int = 50) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV"""
        print(f"\nPerforming hyperparameter tuning using {method} search...")

        # Calculate sample weights
        sample_weight = None
        if hasattr(self, 'class_weights') and self.class_weights:
            weight_map = {self.label_encoder.transform([cls])[0]: weight 
                         for cls, weight in self.class_weights.items()}
            sample_weight = np.array([weight_map[label] for label in y_train])

        # Split for validation
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, 
            stratify=y_train
        )

        # Create base model
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=len(self.label_encoder.classes_),
            tree_method='hist'
        )

        # Setup cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        if method == 'randomized':
            # Use RandomizedSearchCV for faster search
            param_distributions = {
                'n_estimators': randint(200, 500),
                'max_depth': randint(4, 10),
                'learning_rate': uniform(0.01, 0.19),  # 0.01 to 0.2
                'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
                'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
                'min_child_weight': randint(1, 5),
                'gamma': uniform(0, 0.5),
                'reg_alpha': uniform(0, 1.0),
                'reg_lambda': uniform(0.5, 1.5)
            }

            search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring='f1_macro',  # Use F1 for imbalanced data
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=self.random_state
            )
        else:
            # Use GridSearchCV (more thorough but slower)
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 1.5]
            }

            search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='f1_macro',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )

        print(f"Starting {method} search (this may take a while)...")
        search.fit(X_train_fit, y_train_fit, 
                  sample_weight=sample_weight[:len(X_train_fit)] if sample_weight is not None else None)

        self.best_params = search.best_params_
        self.cv_results = search.cv_results_

        print(f"\nBest parameters found: {self.best_params}")
        print(f"Best CV score (F1-macro): {search.best_score_:.4f}")

        # Train model with best parameters using early stopping
        self.model = xgb.XGBClassifier(
            **self.best_params, 
            random_state=self.random_state, 
            n_jobs=-1, 
            objective='multi:softprob', 
            eval_metric='mlogloss',
            num_class=len(self.label_encoder.classes_),
            tree_method='hist'
        )
        
        # Use early_stopping_rounds parameter (compatible with all XGBoost versions)
        self.model.fit(
            X_train_fit, y_train_fit,
            sample_weight=sample_weight[:len(X_train_fit)] if sample_weight is not None else None,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        return self.best_params

    def cross_validation(self, X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5) -> Dict:
        """Perform cross-validation for model evaluation"""
        print(f"\nPerforming {n_folds}-fold cross-validation...")

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # Cross-validation scores
        accuracy_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1_macro')
        precision_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='precision_macro')
        recall_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='recall_macro')

        cv_results = {
            'Accuracy_mean': accuracy_scores.mean(),
            'Accuracy_std': accuracy_scores.std(),
            'F1_mean': f1_scores.mean(),
            'F1_std': f1_scores.std(),
            'Precision_mean': precision_scores.mean(),
            'Precision_std': precision_scores.std(),
            'Recall_mean': recall_scores.mean(),
            'Recall_std': recall_scores.std()
        }

        print("Cross-validation results:")
        print(f"  Accuracy: {cv_results['Accuracy_mean']:.4f} ± {cv_results['Accuracy_std']:.4f}")
        print(f"  F1: {cv_results['F1_mean']:.4f} ± {cv_results['F1_std']:.4f}")
        print(f"  Precision: {cv_results['Precision_mean']:.4f} ± {cv_results['Precision_std']:.4f}")
        print(f"  Recall: {cv_results['Recall_mean']:.4f} ± {cv_results['Recall_std']:.4f}")

        return cv_results

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance on test set"""
        print("\nEvaluating model on test set...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Define all labels
        all_labels = np.arange(len(self.label_encoder.classes_))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Confusion_Matrix': conf_matrix,
            'y_true': y_test,
            'y_pred': y_pred
        }

        print("Test set performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=all_labels, target_names=self.label_encoder.classes_, zero_division=0))
        print("\nConfusion Matrix:")
        print(conf_matrix)

        return metrics

    def plot_feature_importance(self, top_n: int = 10):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return

        plt.figure(figsize=(12, 8))

        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot top N features
        top_features = feature_importance_df.head(top_n)

        plt.subplot(2, 1, 1)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()

        # Plot all features
        plt.subplot(2, 1, 2)
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.title('All Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()

        plt.show()

    def plot_predictions(self, metrics: Dict):
        """Plot confusion matrix"""
        conf_matrix = metrics['Confusion_Matrix']

        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = self.label_encoder.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, X_train: np.ndarray, y_train: np.ndarray):
        """Plot learning curve"""
        from sklearn.model_selection import learning_curve

        plt.figure(figsize=(10, 6))

        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='accuracy',
            n_jobs=-1, random_state=self.random_state
        )

        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Accuracy')
        plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Accuracy')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)

        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def save_model(self, filename: str = 'xgboost_class_model.json'):
        """Save trained model with encoders and metadata"""
        if self.model is None:
            print("No model to save!")
            return

        # Save XGBoost model
        self.model.save_model(filename)
        
        # Save encoders and metadata to a separate file
        metadata_filename = filename.replace('.json', '_metadata.pkl')
        metadata = {
            'label_encoder': self.label_encoder,
            'feature_encoders': self.feature_encoders,
            'feature_names': self.feature_names,
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else None
        }
        
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved as {filename}")
        print(f"Metadata saved as {metadata_filename}")

    def load_model(self, filename: str = 'xgboost_class_model.json'):
        """Load trained model with encoders and metadata"""
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(filename)
        
        # Load metadata
        metadata_filename = filename.replace('.json', '_metadata.pkl')
        if os.path.exists(metadata_filename):
            with open(metadata_filename, 'rb') as f:
                metadata = pickle.load(f)
            self.label_encoder = metadata.get('label_encoder')
            self.feature_encoders = metadata.get('feature_encoders', {})
            self.feature_names = metadata.get('feature_names')
            print(f"Metadata loaded from {metadata_filename}")
        else:
            print(f"Warning: Metadata file not found: {metadata_filename}")
        
        print(f"Model loaded from {filename}")


def main():
    """Main function to run XGBoost training and validation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XGBoost Classification Model Training')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Data file name (e.g., "CAN-HGH_CANHGH_Y")')
    parser.add_argument('--data-dir', type=str, 
                        default='/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data',
                        help='Data directory path (default: data directory)')
    parser.add_argument('--model-dir', type=str,
                        default='/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/nh_work/model_v2_class/xgboost',
                        help='Model save directory path (default: model_v2_class/xgboost)')


    
    args = parser.parse_args()
    
    # Build data file path
    data_filename = args.data
    if not data_filename.endswith('.csv'):
        data_filename ='data_slice_'+args.data + '.csv'
    data_path = os.path.join(args.data_dir, data_filename)
    
    # Build model save path

    model_path = os.path.join(args.model_dir, f'{data_filename}_xgboots_class_model.json')
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("=" * 60)
    print("XGBoost Classification Model Training and Validation")
    print("=" * 60)
    print(f"Data file: {data_path}")
    print(f"Model save path: {model_path}")

    # Initialize trainer
    trainer = XGBoostTrainer(random_state=42)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_path)

    # Train basic model with optimizations
    trainer.train_basic_model(X_train, y_train, use_early_stopping=True, use_class_weights=True)
    basic_metrics = trainer.evaluate_model(X_test, y_test)

    # Hyperparameter tuning (recommended for better performance)
    # Use 'randomized' for faster search or 'grid' for thorough search
    print("\n" + "=" * 60)
    print("Starting hyperparameter tuning...")
    print("=" * 60)
    best_params = trainer.hyperparameter_tuning(X_train, y_train, method='randomized', n_iter=30)
    tuned_metrics = trainer.evaluate_model(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("COMPARISON: Basic vs Tuned Model")
    print("=" * 60)
    print("Basic Model:")
    print(f"  Accuracy: {basic_metrics['Accuracy']:.4f}")
    print(f"  F1: {basic_metrics['F1']:.4f}")
    print("\nTuned Model:")
    print(f"  Accuracy: {tuned_metrics['Accuracy']:.4f}")
    print(f"  F1: {tuned_metrics['F1']:.4f}")
    print(f"  Improvement: {(tuned_metrics['F1'] - basic_metrics['F1']):.4f}")

    # Cross-validation on tuned model
    cv_results = trainer.cross_validation(X_train, y_train)



    # Save model
    trainer.save_model(model_path)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Tuned Model Performance (Best):")
    print(f"  Accuracy: {tuned_metrics['Accuracy']:.4f}")
    print(f"  Precision: {tuned_metrics['Precision']:.4f}")
    print(f"  Recall: {tuned_metrics['Recall']:.4f}")
    print(f"  F1: {tuned_metrics['F1']:.4f}")
    print(f"\nCross-Validation Performance:")
    print(f"  Accuracy: {cv_results['Accuracy_mean']:.4f} ± {cv_results['Accuracy_std']:.4f}")
    print(f"  F1: {cv_results['F1_mean']:.4f} ± {cv_results['F1_std']:.4f}")
    print(f"  Precision: {cv_results['Precision_mean']:.4f} ± {cv_results['Precision_std']:.4f}")
    print(f"  Recall: {cv_results['Recall_mean']:.4f} ± {cv_results['Recall_std']:.4f}")


    # Plot results
    trainer.plot_feature_importance()
    #trainer.plot_predictions(basic_metrics)
    trainer.plot_learning_curve(X_train, y_train)

    return trainer, basic_metrics, cv_results


if __name__ == "__main__":
    trainer, basic_metrics, cv_results = main()
