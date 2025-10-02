#!/usr/bin/env python3
"""
Enterprise Logistic Regression Framework
A minimal, dataset-agnostic implementation for training, evaluating, and deploying classification models.

Design Principles (Zen of Python):
- Simple is better than complex
- Explicit is better than implicit
- Flat is better than nested
- Readability counts
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# ============================================================================
# Configuration - Explicit specification of all experiment parameters
# ============================================================================

@dataclass
class DatasetConfig:
    """Dataset configuration - where and how to load classification data."""
    csv_path: str
    features: List[str]
    label: str
    test_size: float = 0.2
    random_state: int = 42
    transforms: Dict[str, str] = field(default_factory=dict)
    stratify: bool = True  # Stratified train/test split for balanced classes


@dataclass
class ModelConfig:
    """Model architecture and training configuration for classification."""
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 50
    optimizer: str = 'rmsprop'  # rmsprop, adam, sgd
    validation_split: float = 0.2
    threshold: float = 0.5  # Decision threshold for binary classification
    class_weight: Optional[str] = None  # 'balanced' or None


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            name=data['name'],
            dataset=DatasetConfig(**data['dataset']),
            model=ModelConfig(**data['model'])
        )


# ============================================================================
# Data Pipeline - Load, transform, and prepare classification datasets
# ============================================================================

class ClassificationDataset:
    """Dataset abstraction for any tabular classification problem."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_names: Optional[List[str]] = None
        self.n_classes: Optional[int] = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> 'ClassificationDataset':
        """Load CSV data with validation."""
        try:
            self.df = pd.read_csv(self.config.csv_path)
            self.logger.info(f"Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
            self._validate_schema()
            self._encode_labels()
            return self
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _validate_schema(self) -> None:
        """Validate dataset contains required features and label."""
        missing = set(self.config.features + [self.config.label]) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(self.df.columns)}")
        
        # Check for missing values
        null_counts = self.df[self.config.features + [self.config.label]].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Missing values detected:\n{null_counts[null_counts > 0]}")
    
    def _encode_labels(self) -> None:
        """Encode string labels to integers for classification."""
        self.label_encoder = LabelEncoder()
        self.df[self.config.label + '_encoded'] = self.label_encoder.fit_transform(
            self.df[self.config.label]
        )
        self.class_names = self.label_encoder.classes_.tolist()
        self.n_classes = len(self.class_names)
        self.logger.info(f"Encoded {self.n_classes} classes: {self.class_names}")
    
    def transform(self) -> 'ClassificationDataset':
        """Apply feature transformations specified in config."""
        for new_col, expression in self.config.transforms.items():
            try:
                self.df[new_col] = self.df.eval(expression)
                self.logger.info(f"Created feature '{new_col}' = {expression}")
            except Exception as e:
                self.logger.error(f"Transform failed for '{new_col}': {e}")
                raise
        return self
    
    def split(self) -> 'ClassificationDataset':
        """Split into train/test sets with optional stratification."""
        X = self.df[self.config.features].values
        y = self.df[self.config.label + '_encoded'].values
        
        # Stratify to maintain class distribution
        stratify = y if self.config.stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify
        )
        
        # Log class distribution
        train_dist = np.bincount(self.y_train) / len(self.y_train)
        test_dist = np.bincount(self.y_test) / len(self.y_test)
        
        self.logger.info(f"Split dataset: train={len(self.X_train)}, test={len(self.X_test)}")
        self.logger.info(f"Train class distribution: {dict(zip(self.class_names, train_dist))}")
        self.logger.info(f"Test class distribution: {dict(zip(self.class_names, test_dist))}")
        
        return self
    
    def get_class_weights(self) -> Dict[int, float]:
        """Compute class weights for imbalanced datasets."""
        class_counts = np.bincount(self.y_train)
        total = len(self.y_train)
        weights = {i: total / (self.n_classes * count) for i, count in enumerate(class_counts)}
        self.logger.info(f"Computed class weights: {weights}")
        return weights
    
    def describe(self) -> pd.DataFrame:
        """Return descriptive statistics for dataset."""
        return self.df[self.config.features + [self.config.label]].describe(include='all')
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix for numeric features."""
        numeric_features = self.df[self.config.features].select_dtypes(include=[np.number]).columns
        return self.df[list(numeric_features) + [self.config.label + '_encoded']].corr()


# ============================================================================
# Model - Logistic regression with flexible architecture
# ============================================================================

class LogisticRegressionModel:
    """Keras-based logistic regression with automatic binary/multiclass handling."""
    
    def __init__(self, config: ModelConfig, n_features: int, n_classes: int, 
                 feature_names: List[str], class_names: List[str]):
        self.config = config
        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_names = feature_names
        self.class_names = class_names
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self.is_binary = (n_classes == 2)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build(self) -> 'LogisticRegressionModel':
        """Construct logistic regression model architecture."""
        # Input layer for each feature
        inputs = [keras.Input(shape=(1,), name=f'input_{i}') for i in range(self.n_features)]
        
        # Concatenate all inputs
        if len(inputs) > 1:
            concatenated = keras.layers.Concatenate()(inputs)
        else:
            concatenated = inputs[0]
        
        # Output layer with appropriate activation
        if self.is_binary:
            # Binary classification: single sigmoid output
            output = keras.layers.Dense(units=1, activation='sigmoid', name='output')(concatenated)
        else:
            # Multiclass: softmax over all classes
            output = keras.layers.Dense(units=self.n_classes, activation='softmax', name='output')(concatenated)
        
        self.model = keras.Model(inputs=inputs, outputs=output)
        
        # Compile with classification-appropriate loss and metrics
        optimizer = self._get_optimizer()
        loss = 'binary_crossentropy' if self.is_binary else 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.logger.info(
            f"Built {'binary' if self.is_binary else 'multiclass'} model: "
            f"{self.n_features} features → {self.n_classes} classes"
        )
        return self
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get configured optimizer instance."""
        optimizers = {
            'rmsprop': keras.optimizers.RMSprop,
            'adam': keras.optimizers.Adam,
            'sgd': keras.optimizers.SGD
        }
        
        optimizer_class = optimizers.get(self.config.optimizer.lower())
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer_class(learning_rate=self.config.learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              class_weight: Optional[Dict[int, float]] = None) -> 'LogisticRegressionModel':
        """Train model on dataset."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        
        # Split features for multi-input model
        X_split = [X[:, i:i+1] for i in range(self.n_features)]
        
        history = self.model.fit(
            X_split, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            class_weight=class_weight,
            verbose=0
        )
        
        self.history = history.history
        final_acc = self.history['accuracy'][-1]
        final_val_acc = self.history['val_accuracy'][-1]
        
        self.logger.info(
            f"Training complete: Accuracy={final_acc:.4f}, Val_Accuracy={final_val_acc:.4f}"
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions for input features."""
        probas = self.predict_proba(X)
        
        if self.is_binary:
            # Binary: threshold probability at 0.5 (or custom threshold)
            return (probas > self.config.threshold).astype(int).flatten()
        else:
            # Multiclass: argmax over class probabilities
            return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions for input features."""
        if self.model is None:
            raise RuntimeError("Model not built/loaded. Call build() or load() first.")
        
        X_split = [X[:, i:i+1] for i in range(self.n_features)]
        probas = self.model.predict(X_split, verbose=0)
        
        if self.is_binary:
            # Binary: return probability of positive class
            return probas.flatten()
        else:
            # Multiclass: return full probability distribution
            return probas
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model with classification metrics."""
        predictions = self.predict(X)
        probas = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions)
        }
        
        # Compute precision, recall, F1
        average = 'binary' if self.is_binary else 'macro'
        metrics['precision'] = precision_score(y, predictions, average=average, zero_division=0)
        metrics['recall'] = recall_score(y, predictions, average=average, zero_division=0)
        metrics['f1'] = f1_score(y, predictions, average=average, zero_division=0)
        
        # AUC-ROC (if applicable)
        try:
            if self.is_binary:
                metrics['auc_roc'] = roc_auc_score(y, probas)
            else:
                # One-vs-rest AUC for multiclass
                metrics['auc_roc'] = roc_auc_score(y, probas, multi_class='ovr', average='macro')
        except ValueError:
            # May fail if only one class present
            metrics['auc_roc'] = None
        
        self.logger.info(f"Evaluation: {metrics}")
        return metrics
    
    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate confusion matrix for predictions."""
        predictions = self.predict(X)
        return confusion_matrix(y, predictions)
    
    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate detailed classification report."""
        predictions = self.predict(X)
        return classification_report(y, predictions, target_names=self.class_names)
    
    def tune_threshold(self, X: np.ndarray, y: np.ndarray, metric: str = 'f1') -> float:
        """Find optimal decision threshold for binary classification."""
        if not self.is_binary:
            raise ValueError("Threshold tuning only applicable to binary classification")
        
        probas = self.predict_proba(X)
        thresholds = np.linspace(0.1, 0.9, 81)
        
        metric_func = {
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }[metric]
        
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            preds = (probas > threshold).astype(int)
            score = metric_func(y, preds, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
        return best_threshold
    
    def save(self, path: Path, metadata: Dict[str, Any]) -> None:
        """Save model with comprehensive metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(save_dir / 'model.keras')
        
        # Save metadata
        metadata_full = {
            **metadata,
            'model_config': asdict(self.config),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'is_binary': self.is_binary,
            'history': self.history,
            'saved_at': datetime.now().isoformat(),
            'keras_version': keras.__version__,
        }
        
        (save_dir / 'metadata.json').write_text(
            json.dumps(metadata_full, indent=2)
        )
        
        self.logger.info(f"Saved model to {save_dir}")
    
    @classmethod
    def load(cls, path: Path) -> 'LogisticRegressionModel':
        """Load model with metadata."""
        load_dir = Path(path)
        
        # Load metadata
        metadata = json.loads((load_dir / 'metadata.json').read_text())
        
        # Reconstruct config
        config = ModelConfig(**metadata['model_config'])
        
        # Create instance
        instance = cls(
            config=config,
            n_features=metadata['n_features'],
            n_classes=metadata['n_classes'],
            feature_names=metadata['feature_names'],
            class_names=metadata['class_names']
        )
        
        # Load Keras model
        instance.model = keras.models.load_model(load_dir / 'model.keras')
        instance.history = metadata.get('history')
        
        instance.logger.info(f"Loaded model from {load_dir}")
        return instance


# ============================================================================
# Experiment Runner - Orchestrate full classification ML lifecycle
# ============================================================================

class ExperimentRunner:
    """Orchestrate dataset loading, training, evaluation, and persistence."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset: Optional[ClassificationDataset] = None
        self.model: Optional[LogisticRegressionModel] = None
        self.results: Dict[str, Any] = {}
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self) -> 'ExperimentRunner':
        """Execute complete experiment workflow."""
        self.logger.info(f"Starting experiment: {self.config.name}")
        
        # Data pipeline
        self.dataset = (
            ClassificationDataset(self.config.dataset)
            .load()
            .transform()
            .split()
        )
        
        # Get class weights if configured
        class_weight = None
        if self.config.model.class_weight == 'balanced':
            class_weight = self.dataset.get_class_weights()
        
        # Model pipeline
        self.model = (
            LogisticRegressionModel(
                config=self.config.model,
                n_features=len(self.config.dataset.features),
                n_classes=self.dataset.n_classes,
                feature_names=self.config.dataset.features,
                class_names=self.dataset.class_names
            )
            .build()
            .train(self.dataset.X_train, self.dataset.y_train, class_weight=class_weight)
        )
        
        # Evaluation
        train_metrics = self.model.evaluate(self.dataset.X_train, self.dataset.y_train)
        test_metrics = self.model.evaluate(self.dataset.X_test, self.dataset.y_test)
        
        # Confusion matrix
        test_cm = self.model.confusion_matrix(self.dataset.X_test, self.dataset.y_test)
        
        self.results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': test_cm.tolist(),
            'classification_report': self.model.classification_report(
                self.dataset.X_test, self.dataset.y_test
            ),
            'class_names': self.dataset.class_names,
            'dataset_stats': self.dataset.describe().to_dict(),
        }
        
        self.logger.info(
            f"Experiment complete: Test Accuracy = {test_metrics['accuracy']:.4f}, "
            f"F1 = {test_metrics['f1']:.4f}"
        )
        return self
    
    def save(self, output_dir: Path) -> None:
        """Save experiment artifacts."""
        output_path = Path(output_dir) / self.config.name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(output_path / 'config.json')
        
        # Save label encoder
        np.save(output_path / 'label_encoder_classes.npy', self.dataset.label_encoder.classes_)
        
        # Save model
        self.model.save(
            output_path / 'model',
            metadata={
                'experiment_name': self.config.name,
                'dataset_config': asdict(self.config.dataset),
            }
        )
        
        # Save results
        (output_path / 'results.json').write_text(
            json.dumps(self.results, indent=2)
        )
        
        self.logger.info(f"Saved experiment to {output_path}")
    
    @classmethod
    def load(cls, experiment_dir: Path) -> 'ExperimentRunner':
        """Load complete experiment from disk."""
        exp_path = Path(experiment_dir)
        
        # Load config
        config = ExperimentConfig.load(exp_path / 'config.json')
        
        # Create instance
        instance = cls(config)
        
        # Load model
        instance.model = LogisticRegressionModel.load(exp_path / 'model')
        
        # Load results
        if (exp_path / 'results.json').exists():
            instance.results = json.loads((exp_path / 'results.json').read_text())
        
        return instance
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions for batch of inputs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call run() or load() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions for batch of inputs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call run() or load() first.")
        return self.model.predict_proba(X)


# ============================================================================
# CLI Interface - Command-line access to framework
# ============================================================================

def create_example_config(output_path: Path, dataset_type: str = 'churn') -> None:
    """Generate example configuration file for quick start."""
    
    configs = {
        'churn': ExperimentConfig(
            name='customer_churn_classification',
            dataset=DatasetConfig(
                csv_path='churn.csv',
                features=['age', 'tenure', 'monthly_charges'],
                label='churn',
                stratify=True
            ),
            model=ModelConfig(
                learning_rate=0.001,
                epochs=50,
                batch_size=50,
                class_weight='balanced'
            )
        ),
        'fraud': ExperimentConfig(
            name='fraud_detection',
            dataset=DatasetConfig(
                csv_path='transactions.csv',
                features=['amount', 'merchant_category', 'hour_of_day'],
                label='is_fraud',
                stratify=True
            ),
            model=ModelConfig(
                learning_rate=0.001,
                epochs=30,
                class_weight='balanced'
            )
        )
    }
    
    config = configs.get(dataset_type, configs['churn'])
    config.save(output_path)
    print(f"Created example config: {output_path}")


def train_command(args: argparse.Namespace) -> None:
    """Execute training from configuration file."""
    config = ExperimentConfig.load(Path(args.config))
    
    runner = ExperimentRunner(config).run()
    
    if args.output:
        runner.save(Path(args.output))
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT: {config.name}")
    print("="*80)
    print(f"\nTest Accuracy:  {runner.results['test_metrics']['accuracy']:.4f}")
    print(f"Test Precision: {runner.results['test_metrics']['precision']:.4f}")
    print(f"Test Recall:    {runner.results['test_metrics']['recall']:.4f}")
    print(f"Test F1 Score:  {runner.results['test_metrics']['f1']:.4f}")
    if runner.results['test_metrics'].get('auc_roc'):
        print(f"Test AUC-ROC:   {runner.results['test_metrics']['auc_roc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(np.array(runner.results['confusion_matrix']))
    
    print(f"\nClassification Report:")
    print(runner.results['classification_report'])


def predict_command(args: argparse.Namespace) -> None:
    """Generate predictions using trained model."""
    runner = ExperimentRunner.load(Path(args.experiment_dir))
    
    # Load input data
    input_df = pd.read_csv(args.input_csv)
    X = input_df[runner.config.dataset.features].values
    
    # Generate predictions
    predictions = runner.predict_batch(X)
    probabilities = runner.predict_proba(X)
    
    # Map predictions to class names
    predicted_classes = [runner.model.class_names[p] for p in predictions]
    
    # Save predictions
    output_df = input_df.copy()
    output_df['predicted_class'] = predicted_classes
    
    if runner.model.is_binary:
        output_df['probability'] = probabilities
    else:
        for i, class_name in enumerate(runner.model.class_names):
            output_df[f'prob_{class_name}'] = probabilities[:, i]
    
    output_df.to_csv(args.output_csv, index=False)
    
    print(f"Saved predictions to {args.output_csv}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Evaluate model on new dataset."""
    runner = ExperimentRunner.load(Path(args.experiment_dir))
    
    # Load evaluation data
    eval_df = pd.read_csv(args.eval_csv)
    X = eval_df[runner.config.dataset.features].values
    
    # Encode labels using saved encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(
        Path(args.experiment_dir) / 'label_encoder_classes.npy',
        allow_pickle=True
    )
    y = label_encoder.transform(eval_df[runner.config.dataset.label])
    
    # Evaluate
    metrics = runner.model.evaluate(X, y)
    cm = runner.model.confusion_matrix(X, y)
    report = runner.model.classification_report(X, y)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS")
    print("="*80)
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name:20s}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(report)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Enterprise Logistic Regression Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Generate example configuration')
    config_parser.add_argument('--output', required=True, help='Output path for config file')
    config_parser.add_argument('--type', default='churn', choices=['churn', 'fraud'],
                              help='Type of example config')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model from config')
    train_parser.add_argument('--config', required=True, help='Path to experiment config JSON')
    train_parser.add_argument('--output', help='Output directory for experiment artifacts')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--experiment-dir', required=True, help='Path to experiment directory')
    predict_parser.add_argument('--input-csv', required=True, help='Input CSV with features')
    predict_parser.add_argument('--output-csv', required=True, help='Output CSV with predictions')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on new data')
    eval_parser.add_argument('--experiment-dir', required=True, help='Path to experiment directory')
    eval_parser.add_argument('--eval-csv', required=True, help='CSV with features and labels')
    
    args = parser.parse_args()
    
    if args.command == 'create-config':
        create_example_config(Path(args.output), args.type)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
