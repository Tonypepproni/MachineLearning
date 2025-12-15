import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Existing modules
import scrubber 
import linear_regression
# from k_neighbor import k_nei # Commented out if not in use, uncocomment if needed

# New model modules
import naive_bayes
import decision_tree
import xgboost_algo

def main():
    # 1. Load and Scrub Data
    df = pd.read_csv('bank-full.csv', sep=';')
    df = scrubber.scrubber.clean(df)
    
    # 2. Prepare Figure for "One Window" Comparison
    # We create 2 subplots: one for Linear Regression, one for Classification comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4) # Add space between plots

    # --- Plot 1: Linear Regression (Existing Logic) ---
    print("Running Linear Regression...")
    x_vals, y_line = linear_regression.lin_reg_calc.calc(df)
    
    ax1.plot(x_vals, y_line, color='blue', label='Regression Line')
    ax1.set_title("Linear Regression Analysis")
    ax1.set_xlabel("Input Variable")
    ax1.set_ylabel("Target Prediction")
    ax1.legend()

    # --- Plot 2: Classification Model Comparison ---
    print("Running Classification Models...")
    
    # Prepare Classification Data
    # Assuming 'y' is the target. We drop it for X.
    # Note: Ensure your scrubber converts 'y' to numbers (0/1) as seen in your scrubber text
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run Models and Collect Accuracies
    acc_nb = naive_bayes.run(X_train, X_test, y_train, y_test)
    acc_dt = decision_tree.run(X_train, X_test, y_train, y_test)
    acc_xgb = xgboost_algo.run(X_train, X_test, y_train, y_test)

    # Create Bar Chart
    models = ['Naive Bayes', 'Decision Tree', 'XGBoost']
    accuracies = [acc_nb, acc_dt, acc_xgb]
    colors = ['skyblue', 'lightgreen', 'salmon']

    bars = ax2.bar(models, accuracies, color=colors)
    ax2.set_title("Classification Algorithm Accuracy Comparison")
    ax2.set_ylabel("Accuracy Score")
    ax2.set_ylim(0, 1.0)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 1.01*height,
                f'{height:.2%}', ha='center', va='bottom')

    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    main()