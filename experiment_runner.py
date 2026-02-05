import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

# Data Loading
def load_data():
    return pd.read_parquet("data/processed_for_ml.parquet")

# Metric Definitions
# We want to specifically focus on the MINORITY class (0 - Closed) because usually "Open" is dominant.
# Alternatively, balanced accuracy or macro F1.
def report_results(results):
    print("\n" + "="*50)
    print(f" RESULTS SUMMARY (5-Fold CV)")
    print("="*50)
    df_res = pd.DataFrame(results).T
    print(df_res.round(4))

def run_experiments():
    print("Loading data...")
    df = load_data()
    
    X = df.drop(columns=['open'])
    y = df['open']
    
    print(f"Data Loaded. X: {X.shape}, y: {y.shape}")
    print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()}")

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from catboost import CatBoostClassifier
    from imblearn.ensemble import EasyEnsembleClassifier
    from tabpfn import TabPFNClassifier # Small data transformer
    
    # 2. Define Models
    models = {
        "Balanced RF": BalancedRandomForestClassifier(
            n_estimators=200, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=1, 
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            auto_class_weights='Balanced',
            random_state=42,
            allow_writing_files=False
        ),
        "EasyEnsemble": EasyEnsembleClassifier(
            n_estimators=20, # 20 balanced bags
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

    results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        elapsed = time.time() - start
        
        # Aggregate results
        results[name] = {
            'Time (s)': elapsed,
            'ROC AUC': cv_results['test_roc_auc'].mean(),
            'F1 Macro': cv_results['test_f1_macro'].mean(),
            'Precision (Closed)': cv_results['test_precision_closed'].mean(),
            'Recall (Closed)': cv_results['test_recall_closed'].mean(),
            'Balanced Acc': cv_results['test_balanced_acc'].mean()
        }
        
    report_results(results)

if __name__ == "__main__":
    run_experiments()
