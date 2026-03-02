#pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob vaderSentiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Load complete dataset
df_combined = pd.read_csv('/content/complete_feature_dataset.csv')

# Convert labels to binary
df_combined['label_binary'] = (df_combined['label'] == 'AI').astype(int)

# Define all feature columns
STYLOMETRIC_FEATURES = ['char_count', 'word_count', 'avg_word_length']

PUNCTUATION_FEATURES = [
    'exclamation_ratio', 'question_ratio', 'comma_ratio'
]

SENTIMENT_FEATURES = [
    'polarity', 'subjectivity',
    'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu',
    'polarity_variance', 'subjectivity_variance',
    'abs_polarity', 'emotional_engineering_score'
]

POS_FEATURES = []
if 'adj_noun_ratio' in df_combined.columns:
    POS_FEATURES = [
        'adj_count', 'noun_count', 'verb_count', 'adv_count',
        'adj_noun_ratio', 'adj_ratio', 'noun_ratio', 'verb_ratio'
    ]

# Combine all features
ALL_FEATURES = (STYLOMETRIC_FEATURES + PUNCTUATION_FEATURES + SENTIMENT_FEATURES + POS_FEATURES)

# Prepare X and y
X = df_combined[ALL_FEATURES].fillna(0)
y = df_combined['label_binary']

# Setup cross-validation
N_SPLITS = 10
RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

## LOGISTIC REGRESSION
# Scale features
scaler_lr = StandardScaler()
X_scaled_lr = pd.DataFrame(
    scaler_lr.fit_transform(X),
    columns=X.columns
)

# Define model
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    solver='lbfgs',
    C=1.0
)

# Cross-validation
lr_cv_results = cross_validate(
    lr_model, X_scaled_lr, y,
    cv=skf,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Get predictions for confusion matrix and ROC
lr_pred = cross_val_predict(lr_model, X_scaled_lr, y, cv=skf, n_jobs=-1)
lr_proba = cross_val_predict(lr_model, X_scaled_lr, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]

# Train on full data for feature importance
lr_model.fit(X_scaled_lr, y)
feature_importance_lr = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

# Save results
lr_cv_results_df = pd.DataFrame(lr_cv_results)
lr_cv_results_df.to_csv('results_lr.csv', index=False)
feature_importance_lr.to_csv('feature_importance_lr.csv', index=False)
np.save('lr_predictions.npy', lr_pred)
np.save('lr_probabilities.npy', lr_proba)

## RANDOM FOREST
# Random Forest doesn't need scaling, but we'll use original features
# Define model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)

# Cross-validation
rf_cv_results = cross_validate(
    rf_model, X, y,
    cv=skf,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Print results
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    test_scores = rf_cv_results[f'test_{metric}']
    print(f"  {metric.upper()}: {test_scores.mean():.4f} (±{test_scores.std():.4f})")

# Get predictions
rf_pred = cross_val_predict(rf_model, X, y, cv=skf, n_jobs=-1)
rf_proba = cross_val_predict(rf_model, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]

# Train on full data for feature importance
rf_model.fit(X, y)
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Save results
rf_cv_results_df = pd.DataFrame(rf_cv_results)
rf_cv_results_df.to_csv('results_rf.csv', index=False)
feature_importance_rf.to_csv('feature_importance_rf.csv', index=False)
np.save('rf_predictions.npy', rf_pred)
np.save('rf_probabilities.npy', rf_proba)

## SUPPORT VECTOR MACHINE (SVM) Linear Kernel
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Scale features
scaler_svm = StandardScaler()
X_scaled_svm = scaler_svm.fit_transform(X) # dataframe taking too long

# Define model
svm_model = LinearSVC(
  C=1.0,
  random_state=RANDOM_STATE,
  class_weight='balanced',
  max_iter=2000, # for efficiency
  dual='auto' # for efficiency
)

svm_model = CalibratedClassifierCV(svm_model, cv=skf, method='sigmoid')  # wrapped to enable predict_proba

# Cross-validation
svm_cv_results = cross_validate(
  svm_model, X_scaled_svm, y,
  cv=skf,
  scoring=scoring,
  return_train_score=True,
  n_jobs=-1
)

# Get predictions
svm_pred = cross_val_predict(svm_model, X_scaled_svm, y, cv=skf, n_jobs=-1)
svm_proba = cross_val_predict(svm_model, X_scaled_svm, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]

# Save results
pd.DataFrame(svm_cv_results).to_csv('results_svm.csv', index=False)
np.save('svm_predictions.npy', svm_pred)
np.save('svm_probabilities.npy', svm_proba)


## RoBERTa (Transformer)
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Prepare data
df_roberta = df_combined[['text', 'label_binary']].copy()
df_roberta = df_roberta.rename(columns={'label_binary': 'label'})

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Manual cross-validation
roberta_cv_results = {
    'fold': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
}

roberta_all_preds = np.zeros(len(df_roberta))
roberta_all_probs = np.zeros(len(df_roberta))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    # Split data
    train_data = df_roberta.iloc[train_idx].reset_index(drop=True)
    val_data = df_roberta.iloc[val_idx].reset_index(drop=True)

    # Create datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=2,
        problem_type="single_label_classification"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./roberta_fold_{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'./logs_fold_{fold}',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    # Store results
    roberta_cv_results['fold'].append(fold + 1)
    roberta_cv_results['accuracy'].append(eval_results['eval_accuracy'])
    roberta_cv_results['precision'].append(eval_results['eval_precision'])
    roberta_cv_results['recall'].append(eval_results['eval_recall'])
    roberta_cv_results['f1'].append(eval_results['eval_f1'])

    # Get predictions for this fold
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_probs = torch.nn.functional.softmax(
        torch.tensor(predictions.predictions), dim=-1
    )[:, 1].numpy()

    # Store predictions
    roberta_all_preds[val_idx] = pred_labels
    roberta_all_probs[val_idx] = pred_probs

    # Clean up to save memory
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Save results
roberta_cv_results_df = pd.DataFrame(roberta_cv_results)
roberta_cv_results_df.to_csv('results_roberta.csv', index=False)
np.save('roberta_predictions.npy', roberta_all_preds)
np.save('roberta_probabilities.npy', roberta_all_probs)

## FEATURE ABLATION ANALYSIS

# Define feature groups for ablation
feature_groups = {
    'Stylometric': STYLOMETRIC_FEATURES,
    'Punctuation': PUNCTUATION_FEATURES,
    'Sentiment': SENTIMENT_FEATURES,
    'POS': POS_FEATURES
}

ablation_results = []

# Baseline: All features
baseline_f1 = rf_cv_results['test_f1'].mean()
ablation_results.append({
    'Features Removed': 'None (Baseline)',
    'Features Used': 'All',
    'Num Features': len(ALL_FEATURES),
    'F1 Score': baseline_f1,
    'F1 Drop': 0.0
})

def run_rf_cv(X):
    """Train a Random Forest and return mean F1 via cross-validation."""
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    cv_results = cross_validate(rf, X, y, cv=skf, scoring={'f1': 'f1'}, n_jobs=-1)
    return cv_results['test_f1'].mean()

# Ablation: Remove one feature group at a time
print("Ablation Tests (remove one group at a time)")
print('='*60)
for group_name, group_features in feature_groups.items():
    if not group_features:
        continue
    remaining_features = [f for f in ALL_FEATURES if f not in group_features]
    ablated_f1 = run_rf_cv(df_combined[remaining_features].fillna(0))
    f1_drop = baseline_f1 - ablated_f1
    print(f"{group_name:15} | F1: {ablated_f1:.4f} | Drop: {f1_drop:.4f} ({f1_drop/baseline_f1*100:.2f}%)")
    ablation_results.append({
        'Features Removed': group_name,
        'Features Used': f"{len(remaining_features)} features",
        'Num Features': len(remaining_features),
        'F1 Score': ablated_f1,
        'F1 Drop': f1_drop
    })

# Isolation: Use only one feature group at a time
print("\nIsolation Tests (use only one group at a time)")
print('='*60)
for group_name, group_features in feature_groups.items():
    if not group_features:
        continue
    isolated_f1 = run_rf_cv(df_combined[group_features].fillna(0))
    print(f"{group_name:15} | F1: {isolated_f1:.4f} | vs Baseline: {isolated_f1/baseline_f1*100:.2f}%")

# Summary
ablation_df = pd.DataFrame(ablation_results).sort_values('F1 Drop', ascending=False)
print("\nAblation Summary")
print(ablation_df.to_string(index=False))

ablation_df.to_csv('feature_ablation_results.csv', index=False)


## SENSITIVITY ANALYSIS - Random Forest Hyperparameters
from sklearn.model_selection import cross_val_score

# Test different n_estimators
n_estimators_values = [10, 50, 100, 200, 500]
results_n_est = []

for n_est in n_estimators_values:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=5, scoring='f1')
    results_n_est.append({'n_estimators': n_est, 'F1': scores.mean(), 'Std': scores.std()})

# Test different max_depth
max_depth_values = [5, 10, 20, None]
results_depth = []

for depth in max_depth_values:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=5, scoring='f1')
    results_depth.append({'max_depth': str(depth), 'F1': scores.mean(), 'Std': scores.std()})

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# n_estimators sensitivity
df_nest = pd.DataFrame(results_n_est)
axes[0].plot(df_nest['n_estimators'], df_nest['F1'], 'o-', linewidth=2, markersize=8)
axes[0].fill_between(df_nest['n_estimators'],
                      df_nest['F1'] - df_nest['Std'],
                      df_nest['F1'] + df_nest['Std'], alpha=0.3)
axes[0].set_xlabel('Number of Trees', fontsize=12)
axes[0].set_ylabel('F1 Score', fontsize=12)
axes[0].set_title('Sensitivity to n_estimators', fontweight='bold')
axes[0].grid(alpha=0.3)

# max_depth sensitivity
df_depth = pd.DataFrame(results_depth)
axes[1].bar(range(len(df_depth)), df_depth['F1'], yerr=df_depth['Std'], capsize=5)
axes[1].set_xticks(range(len(df_depth)))
axes[1].set_xticklabels(df_depth['max_depth'])
axes[1].set_xlabel('Max Depth', fontsize=12)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Sensitivity to max_depth', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


## TRADEOFF ANALYSIS
from sklearn.metrics import precision_score, recall_score, f1_score
# TRADEOFF 1: Precision vs Recall
thresholds = [0.3, 0.5, 0.7, 0.9]
rf_proba = np.load('rf_probabilities.npy')
y_true = df_combined['label_binary'].values

tradeoff_results = []
for thresh in thresholds:
    y_pred_thresh = (rf_proba >= thresh).astype(int)

    precision = precision_score(y_true, y_pred_thresh)
    recall = recall_score(y_true, y_pred_thresh)
    f1 = f1_score(y_true, y_pred_thresh)

    tradeoff_results.append({
        'Threshold': thresh,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

# Visualize precision-recall tradeoff
df_tradeoff = pd.DataFrame(tradeoff_results)
plt.figure(figsize=(10, 6))
plt.plot(df_tradeoff['Recall'], df_tradeoff['Precision'], 'o-', linewidth=2, markersize=10)
for i, row in df_tradeoff.iterrows():
    plt.annotate(f"θ={row['Threshold']}",
                 (row['Recall'], row['Precision']),
                 textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Recall (Detecting AI)', fontsize=12)
plt.ylabel('Precision (Not Flagging Humans)', fontsize=12)
plt.title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('tradeoff_precision_recall.png', dpi=300, bbox_inches='tight')
plt.show()

# TRADEOFF 2: Training Size vs Accuracy
from sklearn.model_selection import learning_curve

train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
train_sizes_abs, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_mean, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(train_sizes_abs, test_mean, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Learning Curve: Data Size vs Accuracy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('tradeoff_data_size.png', dpi=300, bbox_inches='tight')
plt.show()

# TRADEOFF 3: Model Complexity vs Speed
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
models_speed = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest (10)', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('Random Forest (100)', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM Linear', LinearSVC(random_state=42, max_iter=1000))
]

speed_results = []
for name, model in models_speed:
    start = time.time()
    if 'SVM' in name or 'Logistic' in name:
        scores = cross_val_score(model, X_scaled_lr, y, cv=3, scoring='f1')
    else:
        scores = cross_val_score(model, X, y, cv=3, scoring='f1')
    elapsed = time.time() - start

    speed_results.append({
        'Model': name,
        'F1': scores.mean(),
        'Time (s)': elapsed
    })

df_speed = pd.DataFrame(speed_results)
plt.figure(figsize=(10, 6))
plt.scatter(df_speed['Time (s)'], df_speed['F1'], s=200, alpha=0.6)
for i, row in df_speed.iterrows():
    plt.annotate(row['Model'], (row['Time (s)'], row['F1']),
                 textcoords="offset points", xytext=(5,5), fontsize=9)
plt.xlabel('Training Time (seconds)', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('Speed vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('tradeoff_speed_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()


## Failure Analysis - AI Text Detection
from scipy.stats import ttest_ind

# Load data and predictions
df = pd.read_csv('complete_feature_dataset.csv')
df['predicted_label'] = np.load('rf_predictions.npy')
df['prediction_confidence'] = np.load('rf_probabilities.npy')
df['is_failure'] = df['label_binary'] != df['predicted_label']

# Categorize failure types
failures = df[df['is_failure']].copy()
fp_mask = (failures['label_binary'] == 0) & (failures['predicted_label'] == 1)
fn_mask = (failures['label_binary'] == 1) & (failures['predicted_label'] == 0)
failures['failure_type'] = np.where(fp_mask, 'False Positive (Human→AI)',
                            np.where(fn_mask, 'False Negative (AI→Human)', 'Unknown'))
failures['confidence_error'] = (failures['prediction_confidence'] - failures['label_binary']).abs()

successes = df[~df['is_failure']]

# Helper
KEY_FEATURES = ['subjectivity_variance', 'polarity_variance', 'adj_noun_ratio',
                'word_count', 'exclamation_ratio']

def print_case(label, row):
    print(f"{label} | {row['failure_type']}")
    print(f"Platform: {row['platform']} | Confidence: {row['prediction_confidence']:.3f}")
    print(f"Text: {row['text'][:300]}...")
    print(f"  subjectivity_variance={row['subjectivity_variance']:.6f}, "
          f"polarity_variance={row['polarity_variance']:.6f}, "
          f"adj_noun_ratio={row['adj_noun_ratio']:.3f}, "
          f"word_count={row['word_count']:.0f}")

# Failure Example
example1 = failures[fp_mask].nlargest(1, 'prediction_confidence').iloc[0]
example2 = failures[fn_mask].nsmallest(1, 'prediction_confidence').iloc[0]
borderline = failures[failures['prediction_confidence'].sub(0.5).abs() < 0.1]
example3 = borderline.iloc[0] if len(borderline) > 0 else failures.iloc[2]

print_case("False Positive (Human → AI)", example1)
print_case("False Negative (AI → Human)", example2)
print_case("Borderline / Edge Case", example3)

# Save examples
pd.DataFrame({
    'Example': ['Example 1', 'Example 2', 'Example 3'],
    'Type': [example1['failure_type'], example2['failure_type'], example3['failure_type']],
    'Platform': [example1['platform'], example2['platform'], example3['platform']],
    'Confidence': [example1['prediction_confidence'], example2['prediction_confidence'], example3['prediction_confidence']],
    'Text': [example1['text'][:200], example2['text'][:200], example3['text'][:200]],
    'Subjectivity_Var': [example1['subjectivity_variance'], example2['subjectivity_variance'], example3['subjectivity_variance']],
    'Polarity_Var': [example1['polarity_variance'], example2['polarity_variance'], example3['polarity_variance']]
}).to_csv('failure_examples_detailed.csv', index=False)

# Feature comparison: failures vs successes
print("Feature Comparison: Failures vs Successes")
print(f"{'Feature':<25} {'Failures':>10} {'Successes':>10} {'Diff':>10} {'Sig':>6}")

for feature in KEY_FEATURES:
    fail_vals = failures[feature].dropna()
    succ_vals = successes[feature].dropna()
    diff = fail_vals.mean() - succ_vals.mean()
    _, p_val = ttest_ind(fail_vals, succ_vals)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{feature:<25} {fail_vals.mean():>10.6f} {succ_vals.mean():>10.6f} {diff:>+10.6f} {sig:>6}")

# Failure rate by text length
df['length_category'] = pd.cut(
    df['word_count'],
    bins=[0, 20, 50, 100, 500, 10000],
    labels=['Very Short (0-20)', 'Short (21-50)', 'Medium (51-100)',
            'Long (101-500)', 'Very Long (500+)']
)

print("\nFailure Rate by Text Length")
print(df.groupby('length_category')['is_failure'].agg(['sum', 'mean', 'count'])
        .rename(columns={'sum': 'n_failures', 'mean': 'failure_rate', 'count': 'n_total'})
        .round(4))

# Failure rate by platform
print("\nFailure Rate by Platform")
print(df.groupby('platform')['is_failure'].agg(['sum', 'mean', 'count'])
        .rename(columns={'sum': 'n_failures', 'mean': 'failure_rate', 'count': 'n_total'})
        .round(4))
