import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def random_forest_with_feature_importances(file_path: str, target_column: str, k: int):
    """
    Random Forest with K-Fold Cross-Validation and Feature Importances.

    Args:
        file_path (str): Path to the dataset.
        target_column (str): Name of the target column.
        k (int): Number of folds for Stratified K-Fold Cross-Validation.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
    X = df[selected_features]
    y = df[target_column]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Variables for the best fold
    best_model, best_fold, best_accuracy = None, -1, 0

    # Perform Stratified K-Fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold}:")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Random Forest model
        model = RandomForestClassifier(criterion='gini', random_state=42, n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Update the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_fold = fold

        print(f"  Accuracy: {accuracy:.4f}")

    # Feature importances for the best model
    if best_model:
        importances = best_model.feature_importances_
        feature_importances = pd.Series(importances, index=selected_features).sort_values(ascending=False)

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar', color='skyblue')
        plt.title(f"Feature Importances for Best Random Forest Model (Fold {best_fold})")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("feature_importances_best_fold.png")
        plt.show()

# File path, target column, and number of folds
file_path = 'shuffled_dataset.csv'
target_column = 'class'
k = 10

# Call the function
random_forest_with_feature_importances(file_path, target_column, k)
