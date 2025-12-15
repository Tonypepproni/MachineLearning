from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

def run(X_train, X_test, y_train, y_test):
    print("\n--- Logistic Regression (LR) ---")
    # increasing max_iter to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC Score: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc