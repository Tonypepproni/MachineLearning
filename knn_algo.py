from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

def run(X_train, X_test, y_train, y_test):
    print("\n--- K-Nearest Neighbors (KNN) ---")
    # n_neighbors=5 is a standard starting point
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"AUC Score: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc