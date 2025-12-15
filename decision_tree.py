from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

def run(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree (DT) ---")
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    
    # Get probabilities for the positive class (1)
    # This is required for ROC curves
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate False Positive Rate, True Positive Rate, and AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC Score: {roc_auc:.4f}")
    
    # Return 3 values to match what main.py expects
    return fpr, tpr, roc_auc