# Enterprise Logistic Regression Framework

A minimal, dataset-agnostic implementation for training, evaluating, and deploying classification models. Built for enterprise production with configuration-driven experiments, comprehensive model persistence, and dual API/CLI interfaces.

**Design Philosophy:** Maximal signal, minimal complexity. Single-file implementation (~770 LOC) following Zen of Python principles.

## Features

- **Dataset Agnostic:** Works with any tabular CSV classification problem (customer churn, fraud detection, sentiment analysis)
- **Binary and Multiclass:** Automatic handling of 2-class and N-class problems
- **Configuration Driven:** Reproducible experiments via JSON configs with full parameter tracking
- **Production Ready:** Comprehensive logging, error handling, model serialization with metadata
- **Classification Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix, classification report
- **Dual Interface:** Both programmatic API and CLI for diverse workflows
- **Minimal Dependencies:** Core ML stack only (Keras, TensorFlow, pandas, NumPy, scikit-learn)

## Installation

```bash
git clone https://github.com/davidkimai/google-logistic-regression.git
cd google-logistic-regression
pip install -r requirements.txt
```

**Requirements:** Python 3.9+

## Quick Start

### 1. Generate Example Configuration

```bash
python logistic_regression.py create-config --output churn_config.json --type churn
```

### 2. Train Model

```bash
python logistic_regression.py train --config churn_config.json --output experiments/
```

### 3. Evaluate Results

```bash
python logistic_regression.py evaluate \
    --experiment-dir experiments/customer_churn_classification \
    --eval-csv test_data.csv
```

### 4. Generate Predictions

```bash
python logistic_regression.py predict \
    --experiment-dir experiments/customer_churn_classification \
    --input-csv new_customers.csv \
    --output-csv predictions.csv
```

## Programmatic API

### Training a Binary Classification Model

```python
from logistic_regression import ExperimentConfig, DatasetConfig, ModelConfig, ExperimentRunner
from pathlib import Path

# Define experiment configuration
config = ExperimentConfig(
    name='customer_churn',
    dataset=DatasetConfig(
        csv_path='churn.csv',
        features=['age', 'tenure', 'monthly_charges'],
        label='churn',  # Binary: 0 or 1
        stratify=True  # Maintain class distribution in train/test split
    ),
    model=ModelConfig(
        learning_rate=0.001,
        epochs=50,
        batch_size=50,
        optimizer='rmsprop',
        class_weight='balanced'  # Handle imbalanced classes
    )
)

# Run experiment
runner = ExperimentRunner(config).run()

# Save artifacts
runner.save(Path('experiments/'))

# Inspect results
print(f"Test Accuracy: {runner.results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 Score: {runner.results['test_metrics']['f1']:.4f}")
print(f"Test AUC-ROC:  {runner.results['test_metrics']['auc_roc']:.4f}")
```

### Training a Multiclass Classification Model

```python
config = ExperimentConfig(
    name='sentiment_analysis',
    dataset=DatasetConfig(
        csv_path='reviews.csv',
        features=['text_length', 'exclamation_count', 'positive_words'],
        label='sentiment',  # Multiclass: 'positive', 'neutral', 'negative'
        stratify=True
    ),
    model=ModelConfig(
        learning_rate=0.01,
        epochs=30,
        optimizer='adam'
    )
)

runner = ExperimentRunner(config).run()

# Model automatically handles multiclass (3 classes with softmax activation)
print(f"Number of classes: {runner.dataset.n_classes}")
print(f"Class names: {runner.dataset.class_names}")
```

### Making Predictions

```python
from logistic_regression import ExperimentRunner
import numpy as np

# Load trained model
runner = ExperimentRunner.load(Path('experiments/customer_churn'))

# Prepare input features
X = np.array([[35, 24, 89.50], [52, 48, 120.25]])  # [age, tenure, monthly_charges]

# Class predictions
predictions = runner.predict_batch(X)
print(predictions)  # [0, 1] - class indices

# Probability predictions
probabilities = runner.predict_proba(X)
print(probabilities)  # [0.23, 0.87] - probability of positive class (binary)

# With class names
for i, pred in enumerate(predictions):
    class_name = runner.model.class_names[pred]
    confidence = probabilities[i] if runner.model.is_binary else probabilities[i][pred]
    print(f"Sample {i}: {class_name} (confidence: {confidence:.2f})")
```

### Threshold Tuning (Binary Classification)

```python
# Load validation data
X_val = ...  # Validation features
y_val = ...  # Validation labels

# Find optimal threshold for F1 score
optimal_threshold = runner.model.tune_threshold(X_val, y_val, metric='f1')

# Update model threshold
runner.model.config.threshold = optimal_threshold

# Re-evaluate with new threshold
new_metrics = runner.model.evaluate(X_test, y_test)
print(f"F1 with optimal threshold: {new_metrics['f1']:.4f}")
```

### Confusion Matrix and Classification Report

```python
import numpy as np

# Get confusion matrix
cm = runner.model.confusion_matrix(X_test, y_test)
print("Confusion Matrix:")
print(cm)

# Get detailed classification report
report = runner.model.classification_report(X_test, y_test)
print(report)
```

## Configuration Reference

### DatasetConfig

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `csv_path` | str | Path or URL to CSV file | Required |
| `features` | List[str] | Feature column names | Required |
| `label` | str | Target column name | Required |
| `test_size` | float | Train/test split ratio | 0.2 |
| `random_state` | int | Random seed for reproducibility | 42 |
| `transforms` | Dict[str, str] | Feature engineering expressions | {} |
| `stratify` | bool | Stratified train/test split | True |

### ModelConfig

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `learning_rate` | float | Optimizer learning rate | 0.001 |
| `epochs` | int | Training epochs | 20 |
| `batch_size` | int | Batch size for training | 50 |
| `optimizer` | str | Optimizer (rmsprop, adam, sgd) | rmsprop |
| `validation_split` | float | Validation data ratio | 0.2 |
| `threshold` | float | Decision threshold (binary only) | 0.5 |
| `class_weight` | str | 'balanced' or None | None |

### Feature Transforms

Transforms use pandas `.eval()` syntax for feature engineering:

```json
{
  "transforms": {
    "age_tenure_interaction": "age * tenure",
    "price_per_month": "total_cost / tenure",
    "is_senior": "age > 65"
  }
}
```

## Multi-Dataset Examples

### Customer Churn Prediction (Binary)

```python
config = ExperimentConfig(
    name='churn_prediction',
    dataset=DatasetConfig(
        csv_path='telecom_churn.csv',
        features=['account_length', 'intl_plan', 'voice_messages', 'day_minutes'],
        label='churn',
        stratify=True
    ),
    model=ModelConfig(
        learning_rate=0.001,
        epochs=50,
        class_weight='balanced',  # Handle class imbalance
        threshold=0.5
    )
)

runner = ExperimentRunner(config).run()
print(f"AUC-ROC: {runner.results['test_metrics']['auc_roc']:.4f}")
```

### Fraud Detection (Binary, Highly Imbalanced)

```python
config = ExperimentConfig(
    name='fraud_detection',
    dataset=DatasetConfig(
        csv_path='credit_card_transactions.csv',
        features=['amount', 'hour', 'merchant_category', 'distance_from_home'],
        label='is_fraud',
        stratify=True
    ),
    model=ModelConfig(
        learning_rate=0.01,
        epochs=30,
        class_weight='balanced',  # Critical for imbalanced fraud detection
        threshold=0.3  # Lower threshold to catch more fraud
    )
)

runner = ExperimentRunner(config).run()

# Tune threshold for optimal precision/recall tradeoff
optimal_threshold = runner.model.tune_threshold(
    runner.dataset.X_test, 
    runner.dataset.y_test,
    metric='f1'
)
```

### Sentiment Analysis (Multiclass)

```python
config = ExperimentConfig(
    name='sentiment_classification',
    dataset=DatasetConfig(
        csv_path='product_reviews.csv',
        features=['review_length', 'exclamation_marks', 'positive_word_count', 'negative_word_count'],
        label='sentiment',  # 'positive', 'neutral', 'negative'
        stratify=True
    ),
    model=ModelConfig(
        learning_rate=0.01,
        epochs=40,
        optimizer='adam'
    )
)

runner = ExperimentRunner(config).run()

# Multiclass automatically uses softmax + categorical cross-entropy
print(f"Classes: {runner.dataset.class_names}")
print(f"Per-class F1: {runner.results['classification_report']}")
```

### Loan Default Prediction (Binary)

```python
config = ExperimentConfig(
    name='loan_default',
    dataset=DatasetConfig(
        csv_path='loan_applications.csv',
        features=['credit_score', 'debt_to_income', 'loan_amount', 'employment_years'],
        label='defaulted',
        transforms={'loan_to_income': 'loan_amount / annual_income'}
    ),
    model=ModelConfig(
        learning_rate=0.001,
        epochs=60,
        class_weight='balanced'
    )
)

ExperimentRunner(config).run().save(Path('experiments/'))
```

## Model Persistence

Every experiment saves:
- **model.keras:** Trained Keras model
- **metadata.json:** Full experiment metadata (config, features, class names, metrics, timestamps)
- **config.json:** Experiment configuration for reproducibility
- **results.json:** Training/test metrics, confusion matrix, classification report
- **label_encoder_classes.npy:** Label encoder state for consistent predictions

### Loading Saved Models

```python
runner = ExperimentRunner.load(Path('experiments/customer_churn'))

# Access model
predictions = runner.model.predict(X)
probabilities = runner.model.predict_proba(X)

# Access metadata
class_names = runner.model.class_names
test_accuracy = runner.results['test_metrics']['accuracy']
confusion_matrix = runner.results['confusion_matrix']
```

## Binary vs Multiclass Handling

The framework automatically detects and handles binary vs multiclass classification:

| Aspect | Binary (2 classes) | Multiclass (3+ classes) |
|--------|-------------------|------------------------|
| **Output Layer** | Dense(1, sigmoid) | Dense(n_classes, softmax) |
| **Loss Function** | Binary cross-entropy | Categorical cross-entropy |
| **Predictions** | 0 or 1 | 0 to n_classes-1 |
| **Probabilities** | Single value [0, 1] | Distribution over classes |
| **Threshold** | Tunable (default 0.5) | Argmax (no threshold) |
| **Metrics** | Binary-averaged | Macro-averaged |

## Experiment Comparison

Compare multiple experiments programmatically:

```python
experiments = ['baseline', 'balanced_weights', 'threshold_tuned']
results = []

for exp_name in experiments:
    runner = ExperimentRunner.load(Path(f'experiments/{exp_name}'))
    results.append({
        'name': exp_name,
        'accuracy': runner.results['test_metrics']['accuracy'],
        'f1': runner.results['test_metrics']['f1'],
        'auc': runner.results['test_metrics']['auc_roc']
    })

# Find best model by F1 score
best = max(results, key=lambda x: x['f1'])
print(f"Best model: {best['name']} (F1: {best['f1']:.4f})")
```

## Hyperparameter Tuning

Grid search example:

```python
from itertools import product

learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [32, 50, 100]

best_f1 = 0
best_config = None

for lr, bs in product(learning_rates, batch_sizes):
    config = ExperimentConfig(
        name=f'tune_lr{lr}_bs{bs}',
        dataset=base_dataset_config,
        model=ModelConfig(learning_rate=lr, batch_size=bs, class_weight='balanced')
    )
    
    runner = ExperimentRunner(config).run()
    test_f1 = runner.results['test_metrics']['f1']
    
    if test_f1 > best_f1:
        best_f1 = test_f1
        best_config = config
        
print(f"Best hyperparameters: LR={best_config.model.learning_rate}, BS={best_config.model.batch_size}")
print(f"Best F1: {best_f1:.4f}")
```

## Architecture

**Single-file design** (`logistic_regression.py`) with clear component separation:

```
Configuration Layer (Dataclasses)
├── DatasetConfig: CSV path, features, stratification
├── ModelConfig: Hyperparameters, optimizer, threshold, class weights
└── ExperimentConfig: Complete experiment specification

Data Pipeline (ClassificationDataset)
├── Load: CSV ingestion with validation
├── Encode: Label encoding for string classes
├── Transform: Feature engineering
├── Split: Stratified train/test partitioning
└── Analyze: Statistics and correlations

Model Layer (LogisticRegressionModel)
├── Build: Keras model with sigmoid/softmax
├── Train: Fit with class weights and history tracking
├── Predict: Class predictions (integer labels)
├── Predict_proba: Probability distributions
├── Evaluate: Classification metrics (accuracy, precision, recall, F1, AUC)
├── Confusion Matrix: Per-class prediction breakdown
├── Threshold Tuning: Optimize decision threshold (binary only)
└── Persist: Save/load with metadata

Experiment Runner (ExperimentRunner)
├── Orchestrate: Full classification ML lifecycle
├── Track: Comprehensive metrics and reports
└── Persist: Reproducible artifacts

CLI Interface
├── create-config: Generate example configs
├── train: Execute experiments
├── predict: Batch inference with probabilities
└── evaluate: Model assessment with metrics
```

## Logging

Structured logging for production observability:

```python
2024-10-01 12:00:00 - ClassificationDataset - INFO - Loaded dataset: 10000 rows, 8 columns
2024-10-01 12:00:01 - ClassificationDataset - INFO - Encoded 2 classes: ['no', 'yes']
2024-10-01 12:00:02 - ClassificationDataset - INFO - Split dataset: train=8000, test=2000
2024-10-01 12:00:02 - ClassificationDataset - INFO - Train class distribution: {'no': 0.7, 'yes': 0.3}
2024-10-01 12:00:03 - LogisticRegressionModel - INFO - Built binary model: 4 features → 2 classes
2024-10-01 12:00:15 - LogisticRegressionModel - INFO - Training complete: Accuracy=0.8567, Val_Accuracy=0.8423
2024-10-01 12:00:16 - LogisticRegressionModel - INFO - Evaluation: {'accuracy': 0.845, 'precision': 0.823, 'recall': 0.789, 'f1': 0.805, 'auc_roc': 0.912}
2024-10-01 12:00:17 - ExperimentRunner - INFO - Experiment complete: Test Accuracy = 0.8450, F1 = 0.8050
```

## Error Handling

Comprehensive validation with actionable messages:

```python
# Missing features
ValueError: Missing columns: {'age'}. Available: ['tenure', 'monthly_charges', 'churn']

# Invalid optimizer
ValueError: Unknown optimizer: sgdd. Choose from: rmsprop, adam, sgd

# Threshold tuning on multiclass
ValueError: Threshold tuning only applicable to binary classification

# Transform errors
SyntaxError: Transform failed for 'interaction': name 'age' is not defined
```

## Troubleshooting

**Import errors:** Ensure all dependencies installed: `pip install -r requirements.txt`

**CSV loading fails:** Check network connectivity for URLs, file permissions for local paths

**Class imbalance issues:** Use `class_weight='balanced'` in ModelConfig for imbalanced datasets

**Low F1 score:** Try threshold tuning for binary classification, increase training epochs, adjust learning rate

**AUC is None:** Occurs if only one class present in evaluation set; verify data stratification

**Training divergence:** Reduce learning rate (e.g., 0.0001), normalize features, check for data leakage

## Contributing

This is a minimal reference implementation for classification tasks. For production deployments:
- Add unit tests (pytest framework)
- Implement k-fold cross-validation
- Add regularization (L1/L2/dropout)
- Support categorical feature encoding
- Integrate experiment tracking (MLflow, W&B)
- Implement calibration methods (Platt scaling, isotonic regression)
- Add SHAP/LIME for model interpretability
- Containerize with Docker

## License

Apache License 2.0 (consistent with source material from Google ML Crash Course)

## Citation

Framework inspired by Google ML Crash Course principles, adapted for enterprise classification tasks.

```bibtex
@misc{google-ml-crash-course,
  title={Machine Learning Crash Course},
  author={Google},
  year={2023},
  url={https://developers.google.com/machine-learning/crash-course}
}
```
