import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

def calculate_internal_fidelity_and_plot_pdp(file_path: str, target_column: str, selected_features: list, k: int = 10):
    """
    Calculate Internal Fidelity for the best fold in K-Fold Cross-Validation using the Random Forest Model,
    and plot Partial Dependence Plot (PDP) for selected features.

    Args:
        file_path (str): Path to the dataset.
        target_column (str): Name of the target column.
        selected_features (list): List of features to use for training.
        k (int): Number of folds for Stratified K-Fold Cross-Validation.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    X = df[selected_features].values
    y = df[target_column].values

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    best_model = None
    best_fold = None
    best_accuracy = 0
    X_test_best = None
    y_test_best = None

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train a Random Forest model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        accuracy = model.score(X_test, y_test)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_fold = fold
            X_test_best = X_test
            y_test_best = y_test

    print(f"Best Fold: {best_fold} with Accuracy: {best_accuracy:.4f}")

    # Plot Partial Dependence Plot for 'Hemo' and 'Sc'
    features = [selected_features.index('hemo'), selected_features.index('sc'),
                (selected_features.index('hemo'), selected_features.index('sc'))]

    fig = plt.figure(figsize=(10, 8))  # Increased figure size
    display = PartialDependenceDisplay.from_estimator(
        best_model,
        X,
        features,
        feature_names=selected_features  # Adjusted kind to handle 2-way interactions
    )

    # Customize the plot
    plt.suptitle("Partial Dependence of Hemoglobin (Hemo) and Creatinine (Sc)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Example Usage
file_path = 'shuffled_dataset.csv'
target_column = 'class'
selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']

# Calculate internal fidelity and plot PDP
calculate_internal_fidelity_and_plot_pdp(file_path, target_column, selected_features)
