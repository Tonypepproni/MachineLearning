from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def run(X_train, X_test, y_train, y_test):
    print("\n--- Naive Bayes (NB) ---")
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    # print(classification_report(y_test, y_pred)) # Optional: uncomment for details
    
    return accuracy