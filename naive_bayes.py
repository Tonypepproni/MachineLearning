from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc

def run(X_train, X_test, y_train, y_test):
    print("\n--- Naive Bayes (NB) ---")
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predict probabilities for the positive class (class 1)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC Score: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc