import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Load V2 data
def load_data(version='v2'):
    if version == 'v2':
        return pd.read_parquet("data/processed_for_ml_v2.parquet")
    else:
        return pd.read_parquet("data/processed_for_ml.parquet")

def report_results(results, title="RESULTS SUMMARY (5-Fold CV)"):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)
    df_res = pd.DataFrame(results).T
    print(df_res.round(4))

def run_experiments_v2():
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING EXPERIMENT (V2)")
    print("=" * 80)
    
    # Load both versions for comparison
    print("\nLoading V1 (original delta features)...")
    df_v1 = load_data('v1')
    X_v1 = df_v1.drop(columns=['open'])
    y_v1 = df_v1['open']
    print(f"V1: {X_v1.shape[0]} samples, {X_v1.shape[1]} features")
    
    print("\nLoading V2 (advanced features)...")
    df_v2 = load_data('v2')
    X_v2 = df_v2.drop(columns=['open'])
    y_v2 = df_v2['open']
    print(f"V2: {X_v2.shape[0]} samples, {X_v2.shape[1]} features")
    
    # Models
    models = {
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            auto_class_weights='Balanced',
            random_state=42,
            allow_writing_files=False
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=1,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    }
    
    scoring = {
        'roc_auc': 'roc_auc',
        'f1_macro': 'f1_macro',
        'precision_closed': make_scorer(precision_score, pos_label=0),
        'recall_closed': make_scorer(recall_score, pos_label=0),
        'balanced_acc': 'balanced_accuracy'
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Run V1 experiments
    results_v1 = {}
    print("\n" + "="*80)
    print("TESTING V1 (Original Delta Features)")
    print("="*80)
    
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        cv_results = cross_validate(model, X_v1, y_v1, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start
        
        results_v1[name] = {
            'Time (s)': elapsed,
            'ROC AUC': cv_results['test_roc_auc'].mean(),
            'F1 Macro': cv_results['test_f1_macro'].mean(),
            'Precision (Closed)': cv_results['test_precision_closed'].mean(),
            'Recall (Closed)': cv_results['test_recall_closed'].mean(),
            'Balanced Acc': cv_results['test_balanced_acc'].mean()
        }
    
    report_results(results_v1, "V1 RESULTS (Original Features)")
    
    # Run V2 experiments
    results_v2 = {}
    print("\n" + "="*80)
    print("TESTING V2 (Advanced Features: Interactions + PCA + Category Risk)")
    print("="*80)
    
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        cv_results = cross_validate(model, X_v2, y_v2, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start
        
        results_v2[name] = {
            'Time (s)': elapsed,
            'ROC AUC': cv_results['test_roc_auc'].mean(),
            'F1 Macro': cv_results['test_f1_macro'].mean(),
            'Precision (Closed)': cv_results['test_precision_closed'].mean(),
            'Recall (Closed)': cv_results['test_recall_closed'].mean(),
            'Balanced Acc': cv_results['test_balanced_acc'].mean()
        }
    
    report_results(results_v2, "V2 RESULTS (Advanced Features)")
    
    # Comparison
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS (V2 vs V1)")
    print("="*80)
    
    df_v1_results = pd.DataFrame(results_v1).T
    df_v2_results = pd.DataFrame(results_v2).T
    
    improvement = df_v2_results - df_v1_results
    
    print("\nBalanced Accuracy Improvement:")
    for model in improvement.index:
        delta = improvement.loc[model, 'Balanced Acc']
        v1_score = df_v1_results.loc[model, 'Balanced Acc']
        v2_score = df_v2_results.loc[model, 'Balanced Acc']
        pct_change = (delta / v1_score) * 100
        
        symbol = "🚀" if delta > 0.01 else "✅" if delta > 0 else "⚠️"
        print(f"  {symbol} {model:20s}: {v1_score:.4f} → {v2_score:.4f} ({delta:+.4f}, {pct_change:+.2f}%)")
    
    # Best model
    best_model = df_v2_results['Balanced Acc'].idxmax()
    best_score = df_v2_results.loc[best_model, 'Balanced Acc']
    
    print(f"\n🏆 Best Model (V2): {best_model} with {best_score:.4f} balanced accuracy")
    
    if best_score > 0.68:
        print("   🎯 BREAKTHROUGH! Exceeded the 67% ceiling!")
    elif best_score > df_v1_results['Balanced Acc'].max():
        print(f"   📈 Improved from V1 best: {df_v1_results['Balanced Acc'].max():.4f}")

if __name__ == "__main__":
    run_experiments_v2()
