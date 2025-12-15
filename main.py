import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Existing modules
import scrubber 
import linear_regression

# New model modules
import naive_bayes
import decision_tree
import xgboost_algo

def main():
    # 1. Load and Scrub Data
    print("Loading and cleaning data...")
    df = pd.read_csv('bank-full.csv', sep=';')
    
    # Use your scrubber to clean/map data to numbers
    df = scrubber.scrubber.clean(df)
    
    # 2. Prepare Figure with 2 Subplots
    # figsize=(10, 10) creates a tall window. 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4) # Add space between the two graphs

    # --- Plot 1: Linear Regression ---
    print("Running Linear Regression...")
    try:
        x_vals, y_line = linear_regression.lin_reg_calc.calc(df)
        ax1.plot(x_vals, y_line, color='blue', label='Regression Line')
        ax1.set_title("Linear Regression Analysis")
        ax1.set_xlabel("Input Variable")
        ax1.set_ylabel("Target Prediction")
        ax1.legend()
    except Exception as e:
        print(f"Could not run Linear Regression: {e}")
        ax1.text(0.5, 0.5, "Linear Regression Error", ha='center')

    # --- Plot 2: Classification Performance (ROC Curves) ---
    print("Running Classification Models...")
    
    # Prepare Data
    # 'y' is the target column
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run Models and Get ROC Data (FPR, TPR, AUC)
    fpr_nb, tpr_nb, auc_nb = naive_bayes.run(X_train, X_test, y_train, y_test)
    fpr_dt, tpr_dt, auc_dt = decision_tree.run(X_train, X_test, y_train, y_test)
    fpr_xgb, tpr_xgb, auc_xgb = xgboost_algo.run(X_train, X_test, y_train, y_test)

    # Plot ROC Curves on the second graph (ax2)
    ax2.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.2f})', color='skyblue')
    ax2.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='lightgreen')
    ax2.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', color='salmon')
    
    # Plot diagonal "Random Guess" line
    # ax2.plot([0, 1], [0, 1], 'k--', label='Random Chance')

    ax2.set_title("Model Performance: ROC Curves")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    main()