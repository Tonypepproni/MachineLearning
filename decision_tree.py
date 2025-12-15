from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def run(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree (DT) ---")
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy