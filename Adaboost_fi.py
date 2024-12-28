import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
import joblib

def visualize_feature_importances(file_path: str, target_column: str, selected_features: list, k: int = 10):
    """
    Visualize Feature Importances for the 3rd Fold of AdaBoost using K-Fold Cross-Validation.

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
    X = df[selected_features]
    y = df[target_column]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Perform Stratified K-Fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the AdaBoost model for the 3rd fold
        if fold == 3:
            model = AdaBoostClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save and reload the model to mimic the process
            model_filename = f"adaboost_fold_{fold}.joblib"
            joblib.dump(model, model_filename)
            loaded_model = joblib.load(model_filename)

            # Get feature importances
            feature_importances = loaded_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            # Plot the feature importances
            plt.figure(figsize=(8, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.title("AdaBoost Feature Importances (Fold 3)")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("feature_importances_fold_3_adaboost.png")
            plt.show()

            print("Feature importances plot saved as 'feature_importances_fold_3_adaboost.png'.")
            break

# Example Usage
file_path = 'shuffled_dataset.csv'
target_column = 'class'
selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
visualize_feature_importances(file_path, target_column, selected_features, k=10)
