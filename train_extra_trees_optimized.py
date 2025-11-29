# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import warnings
import os
import json
import joblib
import gc

warnings.filterwarnings('ignore')

class ExtraTreesTrainer:
    """Extra Trees classification model trainer with validation and loss tracking (Optimized for Large Data)"""

    def __init__(self, random_state: int = 42, n_jobs: int = 4):
        """
        Initialize Extra Trees trainer
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs. For 4M rows, suggest 4-8, not -1 (to save RAM overhead).
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.categorical_features = None
        self.training_history = None
        self.model_params = {}

    def prepare_data(self, data_path: str, test_size: float = 0.2, chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and testing data (handles large files and memory optimization)"""
        print("Preparing data...")

        file_size = os.path.getsize(data_path) / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")

        # Read data
        if file_size > 500:
            print(f"Large file detected. Reading in chunks of {chunk_size} rows...")
            chunks = []
            total_rows = 0
            for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows % (chunk_size * 10) == 0:
                    print(f"  Read {total_rows:,} rows...")
            df = pd.concat(chunks, ignore_index=True)
            print(f"Total rows read: {len(df):,}")
        else:
            df = pd.read_csv(data_path)
        
        # Handle missing values
        df_clean = df.dropna()
        
        # Memory Optimization: Free original df
        del df
        gc.collect()

        # Separate features and target
        target_column = df_clean.columns[-1]
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]

        # Identify and Encode categorical features
        categorical_features = []
        label_encoders = {}
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'bool':
                categorical_features.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            elif X[col].dtype in ['int64', 'int32']:
                if X[col].nunique() / len(X) < 0.05:
                    categorical_features.append(col)
        
        self.categorical_features = categorical_features
        self.label_encoders = label_encoders
        print(f"Categorical features: {len(categorical_features)}")

        # Filter rare classes
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index
        if len(rare_classes) > 0:
            print(f"Removing {len(rare_classes)} rare classes")
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]

        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
        self.feature_names = X.columns.tolist()

        # MEMORY OPTIMIZATION: Convert features to float32
        # This saves 50% memory compared to float64
        print("Optimizing memory: Converting features to float32...")
        X = X.astype(np.float32)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def _compute_loss_batched(self, X, y_true, batch_size=50000) -> float:
        """
        Compute log loss in batches WITHOUT storing the full probability matrix.
        This is the key fix for memory leak - we compute loss incrementally.
        """
        n_samples = len(X)
        n_classes = len(self.model.classes_)
        unique_labels = np.unique(y_true)
        
        # Accumulate weighted log loss
        total_loss = 0.0
        total_samples = 0
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_X = X.iloc[i:end] if hasattr(X, 'iloc') else X[i:end]
            batch_y = y_true.iloc[i:end] if hasattr(y_true, 'iloc') else y_true[i:end]
            
            # Predict probabilities for this batch only
            batch_proba = self.model.predict_proba(batch_X)
            
            # Compute batch log loss
            batch_loss = log_loss(batch_y, batch_proba, labels=unique_labels)
            batch_size_actual = end - i
            
            total_loss += batch_loss * batch_size_actual
            total_samples += batch_size_actual
            
            # CRITICAL: Explicitly delete batch arrays to free memory
            del batch_proba
            del batch_X
            del batch_y
        
        # Force garbage collection after processing all batches
        gc.collect()
        
        return total_loss / total_samples if total_samples > 0 else 0.0

    def _predict_proba_batched(self, X, batch_size=50000):
        """Helper to predict probabilities in batches to avoid OOM"""
        n_samples = len(X)
        n_classes = len(self.model.classes_)
        
        # Pre-allocate output array
        y_proba = np.zeros((n_samples, n_classes), dtype=np.float32)
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_X = X.iloc[i:end] if hasattr(X, 'iloc') else X[i:end]
            y_proba[i:end] = self.model.predict_proba(batch_X)
            
            # Explicitly delete batch reference
            del batch_X
            
        gc.collect()
        return y_proba

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   validation_size: float = 0.1, 
                   n_estimators: int = 100,
                   max_depth: Optional[int] = None,
                   min_samples_split: int = 2,
                   min_samples_leaf: int = 1,
                   max_features: str = 'sqrt',
                   verbose: int = 1,
                   track_train_loss: bool = False) -> ExtraTreesClassifier:
        """
        Train Extra Trees classification model using WARM START (Incremental Learning)
        
        Args:
            track_train_loss: If False, skip training loss calculation to save memory.
                             Training loss requires full probability matrix which is expensive.
        """
        print("\nTraining Extra Trees model (Incremental w/ Warm Start)...")

        # Split training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_size, 
            random_state=self.random_state,
            stratify=y_train
        )
        
        print(f"Train split: {len(X_train_split)}, Validation split: {len(X_val)}")

        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'warm_start': True
        }

        # Initialize model ONCE with n_estimators=0
        self.model = ExtraTreesClassifier(
            n_estimators=0,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
            warm_start=True,
            bootstrap=False 
        )

        train_losses = []
        val_losses = []
        iterations_tracked = []
        
        # Define checkpoints (Steps)
        num_checkpoints = min(10, n_estimators)
        step_size = max(1, n_estimators // num_checkpoints)
        checkpoints = list(range(step_size, n_estimators + 1, step_size))
        if checkpoints[-1] != n_estimators:
            checkpoints.append(n_estimators)
        checkpoints = sorted(list(set(checkpoints)))
        
        print(f"Training checkpoints: {checkpoints}")
        
        for checkpoint in checkpoints:
            # Update number of trees
            self.model.n_estimators = checkpoint
            
            # Fit adds new trees (does NOT retrain old ones)
            self.model.fit(X_train_split, y_train_split)
            
            # Validation Tracking using BATCHED loss computation (no full matrix stored)
            try:
                # Compute validation loss incrementally (memory efficient)
                val_loss = self._compute_loss_batched(X_val, y_val, batch_size=50000)
                val_losses.append(val_loss)
                iterations_tracked.append(checkpoint)
                
                # Training loss is optional (expensive for large datasets)
                train_loss = 0.0
                if track_train_loss:
                    train_loss = self._compute_loss_batched(X_train_split, y_train_split, batch_size=100000)
                    train_losses.append(train_loss)
                
                if verbose > 0:
                    if track_train_loss:
                        print(f"  Trees: {checkpoint}/{n_estimators} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                    else:
                        print(f"  Trees: {checkpoint}/{n_estimators} | Val Loss: {val_loss:.5f}")
                    
            except Exception as e:
                print(f"  Warning: Could not calculate loss at step {checkpoint}: {e}")
            
            # CRITICAL: Force garbage collection after each checkpoint
            gc.collect()
        
        # Disable warm_start after training
        self.model.warm_start = False
        
        self.training_history = {
            'train_loss': train_losses if track_train_loss else [],
            'val_loss': val_losses,
            'iterations': iterations_tracked,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0
        }

        return self.model

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance on test set (Batched)"""
        print("\nEvaluating model on test set...")
        
        # Compute test loss incrementally (memory efficient)
        print("Computing test loss (batched)...")
        test_loss = self._compute_loss_batched(X_test, y_test, batch_size=50000)
        
        # Get class predictions in batches
        print("Getting class predictions (batched)...")
        n_samples = len(X_test)
        y_pred = np.zeros(n_samples, dtype=np.int32)
        batch_size = 50000
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_X = X_test.iloc[i:end] if hasattr(X_test, 'iloc') else X_test[i:end]
            batch_proba = self.model.predict_proba(batch_X)
            y_pred[i:end] = self.model.classes_[np.argmax(batch_proba, axis=1)]
            
            # Clean up batch memory
            del batch_proba
            del batch_X
        
        gc.collect()
        
        all_labels = np.arange(len(self.label_encoder.classes_))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', labels=all_labels, zero_division=0)
        
        # Only compute confusion matrix if classes < 50 to avoid clutter
        if len(all_labels) < 50:
            conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)
        else:
            conf_matrix = "Confusion matrix skipped (too many classes)"

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Log_Loss': test_loss,
            'Confusion_Matrix': conf_matrix
        }

        print(f"Test Performance: Acc={accuracy:.4f}, F1={f1:.4f}, Loss={test_loss:.4f}")
        return metrics

    def plot_training_loss(self, save_path: str = None):
        """Plot training and validation loss curves"""
        if not self.training_history or not self.training_history['val_loss']:
            print("No history to plot.")
            return
            
        plt.figure(figsize=(10, 6))
        iters = self.training_history['iterations']
        
        if self.training_history['train_loss']:
            plt.plot(iters, self.training_history['train_loss'], 'b-o', label='Train Loss')
        plt.plot(iters, self.training_history['val_loss'], 'r-s', label='Val Loss')
        
        plt.xlabel('Number of Estimators')
        plt.ylabel('Log Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        if self.model is None: return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Importance plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def save_model(self, filename: str = 'extra_trees_model.pkl'):
        """Save model"""
        if self.model is None: return
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        print(f"Saving model to {filename}...")
        # Compress=3 saves disk space significantly for large Random Forests
        joblib.dump(self.model, filename, compress=3)
        
        # Save meta-data
        info = {
            'classes': self.label_encoder.classes_.tolist(),
            'feature_names': self.feature_names,
            'params': self.model_params
        }
        with open(filename.replace('.pkl', '_info.json'), 'w') as f:
            json.dump(info, f, default=str)
        print("Model saved.")


def main():
    print("=" * 60)
    print("Extra Trees Large Scale Training (Memory Optimized)")
    print("=" * 60)

    # Use n_jobs=4 to balance speed and memory overhead
    trainer = ExtraTreesTrainer(random_state=42, n_jobs=4)

    # UPDATE PATH HERE
    data_path = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data/all_route_data_v2.csv'
    
    # 1. Prepare Data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_path)

    # 2. Train Model
    # MEMORY OPTIMIZATION PARAMS:
    # - min_samples_leaf=10: Reduces model size significantly (fewer nodes)
    # - max_depth=30: Prevents infinitely deep trees
    # - n_estimators=2000: Trained incrementally
    # - track_train_loss=False: Skip train loss to save memory (set True if needed)
    trainer.train_model(
        X_train, y_train,
        validation_size=0.1,
        n_estimators=2000, 
        max_depth=30,  
        min_samples_split=10, 
        min_samples_leaf=10,
        max_features='sqrt',
        verbose=1,
        track_train_loss=False  # Set to True if you need train loss curve
    )

    # 3. Evaluate
    metrics = trainer.evaluate_model(X_test, y_test)

    # 4. Save
    model_dir = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/nh_work/model_alldata_v2_class/v1.1/extra_trees'
    trainer.save_model(os.path.join(model_dir, 'extra_trees_model.pkl'))
    
    trainer.plot_training_loss(os.path.join(model_dir, 'loss_curve.png'))
    trainer.plot_feature_importance(20, os.path.join(model_dir, 'feat_imp.png'))

if __name__ == "__main__":
    main()
