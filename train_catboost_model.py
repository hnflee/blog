# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import warnings
import os
warnings.filterwarnings('ignore')


class CatBoostTrainer:
    """CatBoost classification model trainer with validation and hyperparameter tuning"""

    def __init__(self, random_state: int = 42, use_gpu: bool = True, gpu_device_ids: str = None):
        """
        Initialize CatBoost trainer
        
        Args:
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU if available (default: True)
            gpu_device_ids: GPU device IDs to use, e.g., '0' or '0,1' for multiple GPUs (default: None, uses all available)
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.gpu_device_ids = gpu_device_ids
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.cv_results = None
        self.label_encoder = None
        self.categorical_features = None
        self.model_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1
        }
        self.training_history = None  # Store training and validation loss history
        
        # Check GPU availability
        self.task_type = self._check_gpu_availability()
        print(f"Training will use: {self.task_type}")

    def prepare_data(self, data_path: str, test_size: float = 0.2, chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and testing data (handles large files with chunking)"""
        print("Preparing data...")

        # Check file size and use chunking if needed
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")

        # Read data in chunks if file is large
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
        
        print(f"Original dataset shape: {df.shape}")

        # Handle missing values
        df_clean = df.dropna()
        print(f"After removing missing values: {df_clean.shape}")

        # Separate features and target
        target_column = df_clean.columns[-1]
        X = df_clean.drop(target_column, axis=1)
        y = df_clean[target_column]

        # Identify categorical features (object, bool, and string types)
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'bool':
                categorical_features.append(col)
            # Also check if numeric column has low cardinality (might be categorical)
            elif X[col].dtype in ['int64', 'int32']:
                unique_ratio = X[col].nunique() / len(X)
                if unique_ratio < 0.05:  # Less than 5% unique values
                    categorical_features.append(col)
        
        self.categorical_features = categorical_features
        print(f"Identified {len(categorical_features)} categorical features: {categorical_features[:10]}...")  # Show first 10

        # Filter out classes with fewer than 2 samples
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index
        if len(rare_classes) > 0:
            print(f"Removing {len(rare_classes)} classes with fewer than 2 samples")
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]

        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)

        self.feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target classes: {len(self.label_encoder.classes_)} classes")
        print(f"Target classes (first 10): {list(self.label_encoder.classes_[:10])}...")

        return X_train, X_test, y_train, y_test

    def _check_gpu_availability(self) -> str:
        """Check if GPU is available for CatBoost"""
        if not self.use_gpu:
            return 'CPU'
        
        try:
            # Method 1: Try to get GPU device count from catboost.utils
            try:
                from catboost.utils import get_gpu_device_count
                gpu_count = get_gpu_device_count()
                if gpu_count > 0:
                    print(f"✓ GPU available: {gpu_count} GPU(s) detected via CatBoost utils")
                    return 'GPU'
            except (ImportError, AttributeError):
                pass
            
            # Method 2: Check CUDA availability via subprocess (nvidia-smi)
            import subprocess
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    gpu_count = len(result.stdout.strip().split('\n'))
                    print(f"✓ GPU available: {gpu_count} GPU(s) detected via nvidia-smi")
                    return 'GPU'
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            print("⚠ GPU requested but not available, falling back to CPU")
            return 'CPU'
        except Exception as e:
            print(f"⚠ GPU check failed: {e}, falling back to CPU")
            return 'CPU'
    
    def _get_device_params(self) -> Dict:
        """Get device-specific parameters for CatBoost"""
        params = {
            'task_type': self.task_type,
        }
        
        if self.task_type == 'GPU':
            # GPU-specific parameters
            if self.gpu_device_ids is not None:
                params['devices'] = self.gpu_device_ids
            # GPU training benefits from larger batch sizes
            params['max_bin'] = 254  # Maximum value for GPU
            # GPU can handle more iterations efficiently
            if 'iterations' in self.model_params and self.model_params['iterations'] < 500:
                # Suggest more iterations for GPU
                pass  # Keep user-specified iterations
        
        return params

    def get_categorical_indices(self, X: pd.DataFrame) -> List[int]:
        """Get indices of categorical features for CatBoost"""
        if self.categorical_features is None:
            return []
        categorical_indices = [X.columns.get_loc(col) for col in self.categorical_features if col in X.columns]
        return categorical_indices

    def train_basic_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         validation_size: float = 0.1) -> cb.CatBoostClassifier:
        """
        Train basic CatBoost classification model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_size: Fraction of training data to use as validation set (default: 0.1)
        """
        print("\nTraining basic CatBoost model...")

        categorical_indices = self.get_categorical_indices(X_train)

        # Split training data into train and validation sets
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_size, 
            random_state=self.random_state,
            stratify=y_train
        )
        
        print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}")

        self.model_params = {
            'iterations': 80000,
            'depth': 6,
            'learning_rate': 0.1,
            'early_stopping_rounds': 500
        }
        
        # Get device-specific parameters
        device_params = self._get_device_params()
        
        self.model = cb.CatBoostClassifier(
            iterations=self.model_params['iterations'],
            depth=self.model_params['depth'],
            learning_rate=self.model_params['learning_rate'],
            loss_function='MultiClass',
            random_seed=self.random_state,
            verbose=100,
            cat_features=categorical_indices,
            thread_count=-1 if self.task_type == 'CPU' else None,  # GPU doesn't need thread_count
            early_stopping_rounds=self.model_params['early_stopping_rounds'],
            **device_params
        )

        # Train with validation set
        self.model.fit(
            X_train_split, y_train_split,
            eval_set=(X_val, y_val),
            cat_features=categorical_indices,
            verbose=100,
            plot=False  # We'll plot separately
        )
        
        # Get training and validation loss history
        evals_result = self.model.get_evals_result()
        
        # Print training and validation loss
        print("\n" + "=" * 60)
        print("Training and Validation Loss")
        print("=" * 60)
        
        if evals_result:
            # CatBoost typically stores results with metric names as keys
            # For MultiClass, the metric is usually 'MultiClass' or 'MultiClass:loss'
            for metric_name, metric_data in evals_result.items():
                train_values = None
                val_values = None
                
                # Extract training and validation values
                if 'learn' in metric_data:
                    train_dict = metric_data['learn']
                    # Try different possible key names
                    for key in train_dict.keys():
                        if isinstance(train_dict[key], list):
                            train_values = train_dict[key]
                            break
                
                if 'validation' in metric_data:
                    val_dict = metric_data['validation']
                    # Try different possible key names
                    for key in val_dict.keys():
                        if isinstance(val_dict[key], list):
                            val_values = val_dict[key]
                            break
                
                # Print training loss
                if train_values:
                    final_train_loss = train_values[-1]
                    best_train_loss = min(train_values)
                    best_train_iter = train_values.index(best_train_loss) + 1
                    print(f"\nTraining Loss ({metric_name}):")
                    print(f"  Final: {final_train_loss:.6f}")
                    print(f"  Best:  {best_train_loss:.6f} (iteration {best_train_iter})")
                    print(f"  First: {train_values[0]:.6f}")
                
                # Print validation loss
                if val_values:
                    final_val_loss = val_values[-1]
                    best_val_loss = min(val_values)
                    best_val_iter = val_values.index(best_val_loss) + 1
                    print(f"\nValidation Loss ({metric_name}):")
                    print(f"  Final: {final_val_loss:.6f}")
                    print(f"  Best:  {best_val_loss:.6f} (iteration {best_val_iter})")
                    print(f"  First: {val_values[0]:.6f}")
                    
                    # Show early stopping info
                    best_iter = self.model.get_best_iteration()
                    if best_iter:
                        print(f"  Early stopping at iteration: {best_iter}")
                    
                    # Show improvement
                    if len(val_values) > 1:
                        improvement = ((val_values[0] - best_val_loss) / val_values[0]) * 100
                        print(f"  Improvement: {improvement:.2f}%")
                
                # Store training history for plotting (save the first metric found)
                if not self.training_history:
                    self.training_history = {
                        'train_loss': train_values if train_values else [],
                        'val_loss': val_values if val_values else [],
                        'metric_name': metric_name
                    }
        
        print("\nBasic model training completed.")

        return self.model

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 20) -> Dict:
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        print("\nPerforming hyperparameter tuning...")

        categorical_indices = self.get_categorical_indices(X_train)

        # Define parameter grid for CatBoost
        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
            'random_strength': [0, 1],
            'bagging_temperature': [0, 1]
        }

        # Get device-specific parameters
        device_params = self._get_device_params()
        
        # Create base model
        catboost_model = cb.CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=self.random_state,
            verbose=False,
            cat_features=categorical_indices,
            thread_count=-1 if self.task_type == 'CPU' else None,
            **device_params
        )

        # Setup cross-validation
        cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)  # Reduced to 3 folds for speed

        # Randomized search with cross-validation (faster than GridSearch)
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            estimator=catboost_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Number of parameter settings sampled
            scoring='accuracy',
            cv=cv,
            verbose=1,
            n_jobs=1,  # CatBoost already uses all threads
            random_state=self.random_state
        )

        print("Starting RandomizedSearchCV (this may take a while)...")
        random_search.fit(X_train, y_train, cat_features=categorical_indices, verbose=False)

        self.best_params = random_search.best_params_
        self.cv_results = random_search.cv_results_

        print(f"\nBest parameters found: {self.best_params}")
        print(f"Best CV score (Accuracy): {random_search.best_score_:.4f}")

        # Update model parameters with best params
        self.model_params.update(self.best_params)
        
        # Get device-specific parameters
        device_params = self._get_device_params()
        
        # Train model with best parameters
        self.model = cb.CatBoostClassifier(
            **self.best_params,
            loss_function='MultiClass',
            random_seed=self.random_state,
            verbose=100,
            cat_features=categorical_indices,
            thread_count=-1 if self.task_type == 'CPU' else None,
            **device_params
        )
        self.model.fit(X_train, y_train, cat_features=categorical_indices, verbose=100)

        return self.best_params

    def cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series, n_folds: int = 5) -> Dict:
        """Perform cross-validation for model evaluation"""
        print(f"\nPerforming {n_folds}-fold cross-validation...")

        categorical_indices = self.get_categorical_indices(X_train)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        # Manual cross-validation to support categorical features
        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            print(f"  Processing fold {fold_idx + 1}/{n_folds}...")
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Get device-specific parameters
            device_params = self._get_device_params()
            
            # Create a new model for this fold using saved model parameters
            fold_model = cb.CatBoostClassifier(
                iterations=self.model_params.get('iterations', 100),
                depth=self.model_params.get('depth', 6),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                loss_function='MultiClass',
                random_seed=self.random_state,
                verbose=False,
                cat_features=categorical_indices,
                thread_count=-1 if self.task_type == 'CPU' else None,
                **device_params
            )

            fold_model.fit(X_fold_train, y_fold_train, cat_features=categorical_indices, verbose=False)
            
            # Use CPU for prediction (faster for inference)
            y_pred = fold_model.predict(X_fold_val, task_type='CPU', thread_count=-1)

            accuracy_scores.append(accuracy_score(y_fold_val, y_pred))
            f1_scores.append(f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
            precision_scores.append(precision_score(y_fold_val, y_pred, average='macro', zero_division=0))
            recall_scores.append(recall_score(y_fold_val, y_pred, average='macro', zero_division=0))

        accuracy_scores = np.array(accuracy_scores)
        f1_scores = np.array(f1_scores)
        precision_scores = np.array(precision_scores)
        recall_scores = np.array(recall_scores)

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

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, chunk_size: int = 50000) -> Dict:
        """
        Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            chunk_size: Number of samples to process at a time for large datasets (default: 50000)
        """
        print("\nEvaluating model on test set...")
        print(f"Test set size: {len(X_test):,} samples")

        # Always use CPU for prediction - it's often faster and more stable than GPU for inference
        # GPU is better for training but CPU is typically better for inference
        predict_task_type = 'CPU'
        print(f"Using {predict_task_type} for prediction (CPU is typically faster for inference)")

        # Make predictions with chunking for large datasets
        if len(X_test) > chunk_size:
            print(f"Large test set detected. Processing in chunks of {chunk_size:,} samples...")
            y_pred_list = []
            n_chunks = (len(X_test) + chunk_size - 1) // chunk_size
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(X_test))
                X_chunk = X_test.iloc[start_idx:end_idx]
                
                print(f"  Processing chunk {i+1}/{n_chunks} (samples {start_idx:,} to {end_idx:,})...")
                # Specify task_type='CPU' and thread_count for faster inference
                chunk_pred = self.model.predict(X_chunk, task_type=predict_task_type, thread_count=-1)
                y_pred_list.append(chunk_pred)
            
            print(f"  Completed all {n_chunks} chunks. Combining predictions...")
            y_pred = np.concatenate(y_pred_list).ravel()
        else:
            print("Making predictions on test set...")
            # Specify task_type='CPU' and thread_count for faster inference
            y_pred = self.model.predict(X_test, task_type=predict_task_type, thread_count=-1).ravel()
        
        print("Predictions completed.")

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
        print(classification_report(
            y_test, y_pred, 
            labels=all_labels, 
            target_names=self.label_encoder.classes_, 
            zero_division=0
        ))
        print("\nConfusion Matrix:")
        print(conf_matrix)

        return metrics

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        if self.model is None:
            print("Model not trained yet!")
            return

        plt.figure(figsize=(12, 10))

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
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Plot all features
        plt.subplot(2, 1, 2)
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.title('All Feature Importance')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()

    def plot_confusion_matrix(self, metrics: Dict, save_path: str = None):
        """Plot confusion matrix"""
        # --- 修改开始：先获取数据并展平为一维数组 ---
        # .ravel() 或 .flatten() 可以将 (N, 1) 变为 (N,)
        y_true = np.array(metrics['y_true']).ravel()
        y_pred = np.array(metrics['y_pred']).ravel()
        # --- 修改结束 ---

        # 使用展平后的数据计算混淆矩阵逻辑
        # Limit the number of classes shown if too many
        max_classes = 20
        if len(self.label_encoder.classes_) > max_classes:
            print(f"Too many classes ({len(self.label_encoder.classes_)}). Showing top {max_classes} classes by frequency.")
            
            # Get top classes by frequency in test set
            # 使用展平后的 y_true
            class_counts = pd.Series(y_true).value_counts()
            top_classes = class_counts.head(max_classes).index
            
            # 计算 mask，现在两边都是 (N,)，不会发生广播，结果也是 (N,)
            mask = np.isin(y_true, top_classes) & np.isin(y_pred, top_classes)
            
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
            classes_to_show = self.label_encoder.classes_[top_classes]
        else:
            conf_matrix = metrics['Confusion_Matrix']
            classes_to_show = self.label_encoder.classes_

        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(classes_to_show))
        plt.xticks(tick_marks, classes_to_show, rotation=45, ha='right')
        plt.yticks(tick_marks, classes_to_show)

        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()

    def plot_training_loss(self, save_path: str = None):
        """Plot training and validation loss curves"""
        if self.training_history is None or (not self.training_history.get('train_loss') and not self.training_history.get('val_loss')):
            print("No training history available to plot!")
            return
        
        train_loss = self.training_history.get('train_loss', [])
        val_loss = self.training_history.get('val_loss', [])
        metric_name = self.training_history.get('metric_name', 'Loss')
        
        if not train_loss and not val_loss:
            print("No training history data to plot!")
            return
        
        plt.figure(figsize=(12, 6))
        
        if train_loss and len(train_loss) > 0:
            plt.plot(train_loss, label=f'Training {metric_name}', color='blue', alpha=0.7, linewidth=2)
        
        if val_loss and len(val_loss) > 0:
            plt.plot(val_loss, label=f'Validation {metric_name}', color='red', alpha=0.7, linewidth=2)
            
            # Mark best validation point
            best_iter = val_loss.index(min(val_loss))
            best_val_loss = min(val_loss)
            plt.axvline(x=best_iter, color='green', linestyle='--', alpha=0.5, linewidth=1)
            plt.plot(best_iter, best_val_loss, 'go', markersize=8, label=f'Best Val: {best_val_loss:.6f}')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(f'{metric_name}', fontsize=12)
        plt.title('Training and Validation Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training loss plot saved to {save_path}")
        else:
            plt.show()

    def save_model(self, filename: str = 'catboost_class_model.cbm'):
        """Save trained model"""
        if self.model is None:
            print("No model to save!")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        self.model.save_model(filename)
        print(f"Model saved as {filename}")

        # Also save label encoder info
        import json
        label_info = {
            'classes': self.label_encoder.classes_.tolist(),
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'model_params': self.model_params
        }
        info_filename = filename.replace('.cbm', '_info.json')
        with open(info_filename, 'w') as f:
            json.dump(label_info, f, indent=2)
        print(f"Model info saved as {info_filename}")

    def load_model(self, filename: str = 'catboost_class_model.cbm'):
        """Load trained model"""
        self.model = cb.CatBoostClassifier()
        self.model.load_model(filename)
        print(f"Model loaded from {filename}")

        # Load label encoder info
        import json
        info_filename = filename.replace('.cbm', '_info.json')
        if os.path.exists(info_filename):
            with open(info_filename, 'r') as f:
                label_info = json.load(f)
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(label_info['classes'])
            self.categorical_features = label_info.get('categorical_features', [])
            self.feature_names = label_info.get('feature_names', [])
            # Restore task type and model params if available
            if 'task_type' in label_info:
                self.task_type = label_info['task_type']
            if 'model_params' in label_info:
                self.model_params = label_info['model_params']


def main():
    """
    Main function to run CatBoost training and validation
    
    GPU Usage:
    - By default, GPU will be automatically detected and used if available
    - To force CPU training: use_gpu=False
    - To specify GPU devices: gpu_device_ids='0' or '0,1' for multiple GPUs
    - GPU training is significantly faster for large datasets
    
    Example:
        # Use GPU automatically (default)
        trainer = CatBoostTrainer(random_state=42, use_gpu=True)
        
        # Force CPU training
        trainer = CatBoostTrainer(random_state=42, use_gpu=False)
        
        # Use specific GPU device
        trainer = CatBoostTrainer(random_state=42, use_gpu=True, gpu_device_ids='0')
    """
    print("=" * 60)
    print("CatBoost Classification Model Training and Validation")
    print("=" * 60)

    # Initialize trainer with GPU support (automatically detects and uses GPU if available)
    # Set use_gpu=False to force CPU training
    # Set gpu_device_ids='0' or '0,1' to specify which GPUs to use
    trainer = CatBoostTrainer(random_state=42, use_gpu=True, gpu_device_ids=None)

    # Prepare data
    data_path = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/data/all_route_data.csv'
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_path)

    # Train basic model
    trainer.train_basic_model(X_train, y_train)
    basic_metrics = trainer.evaluate_model(X_test, y_test)

    # Hyperparameter tuning (commented out for faster execution)
    # Uncomment the following line for hyperparameter tuning
    # best_params = trainer.hyperparameter_tuning(X_train, y_train, n_iter=10)
    # tuned_metrics = trainer.evaluate_model(X_test, y_test)

    # Cross-validation
    cv_results = trainer.cross_validation(X_train, y_train, n_folds=5)

    # Save model
    model_dir = '/Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/nh_work/model_alldata_v2_class/catboost'
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(os.path.join(model_dir, 'catboost_class_model.cbm'))

    # Plot results
    trainer.plot_training_loss(
        save_path=os.path.join(model_dir, 'catboost_training_loss.png')
    )
    trainer.plot_feature_importance(
        top_n=20, 
        save_path=os.path.join(model_dir, 'catboost_feature_importance.png')
    )
    trainer.plot_confusion_matrix(
        basic_metrics,
        save_path=os.path.join(model_dir, 'catboost_confusion_matrix.png')
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Basic Model Performance:")
    print(f"  Accuracy: {basic_metrics['Accuracy']:.4f}")
    print(f"  Precision: {basic_metrics['Precision']:.4f}")
    print(f"  Recall: {basic_metrics['Recall']:.4f}")
    print(f"  F1: {basic_metrics['F1']:.4f}")
    print(f"\nCross-Validation Performance:")
    print(f"  Accuracy: {cv_results['Accuracy_mean']:.4f} ± {cv_results['Accuracy_std']:.4f}")
    print(f"  F1: {cv_results['F1_mean']:.4f} ± {cv_results['F1_std']:.4f}")
    print(f"  Precision: {cv_results['Precision_mean']:.4f} ± {cv_results['Precision_std']:.4f}")
    print(f"  Recall: {cv_results['Recall_mean']:.4f} ± {cv_results['Recall_std']:.4f}")

    return trainer, basic_metrics, cv_results


if __name__ == "__main__":
    trainer, basic_metrics, cv_results = main()
