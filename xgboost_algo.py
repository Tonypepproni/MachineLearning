from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def run(X_train, X_test, y_train, y_test):
    print("\n--- XGBoost (XGB) ---")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy