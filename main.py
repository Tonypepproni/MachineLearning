import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your scrubber
import scrubber 

# Import all 6 Classification Algorithms
import naive_bayes
import decision_tree
import xgboost_algo
import logistic_regression_algo
import knn_algo
import random_forest_algo

def main():
    print("Loading and cleaning data...")
    df = pd.read_csv('bank-full.csv', sep=';')
    
    # Clean data using your scrubber
    df = scrubber.scrubber.clean(df)
    
    # Prepare Data
    # Assuming 'y' is the target column mapped to 0/1
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split Data (80% Train, 20% Test)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Run Classification Models ---
    print("Running Classification Models...")

    # 1. Naive Bayes
    fpr_nb, tpr_nb, auc_nb = naive_bayes.run(X_train, X_test, y_train, y_test)
    
    # 2. Decision Tree
    fpr_dt, tpr_dt, auc_dt = decision_tree.run(X_train, X_test, y_train, y_test)
    
    # 3. XGBoost
    fpr_xgb, tpr_xgb, auc_xgb = xgboost_algo.run(X_train, X_test, y_train, y_test)
    
    # 4. Logistic Regression
    fpr_lr, tpr_lr, auc_lr = logistic_regression_algo.run(X_train, X_test, y_train, y_test)
    
    # 5. K-Nearest Neighbors
    fpr_knn, tpr_knn, auc_knn = knn_algo.run(X_train, X_test, y_train, y_test)
    
    # 6. Random Forest
    fpr_rf, tpr_rf, auc_rf = random_forest_algo.run(X_train, X_test, y_train, y_test)

    # --- Plotting ---
    print("Generating Plot...")
    plt.figure(figsize=(12, 10))
    
    # Plot ROC Curves
    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.2f})', color='skyblue')
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='lightgreen')
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', color='salmon')
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})', color='orange')
    plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', color='purple')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='darkblue')

    # Graph Formatting
    plt.title("Model Performance Comparison: ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()