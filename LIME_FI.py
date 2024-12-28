import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
import joblib
import re  # For extracting base feature names

def calculate_lime_feature_importances(file_path: str, target_column: str, selected_features: list, k: int = 10):
    """
    Calculate Global LIME Feature Importances for the 3rd Fold of AdaBoost using K-Fold Cross-Validation.

    Args:
        file_path (str): Path to the dataset.
        target_column (str): Name of the target column.
        selected_features (list): List of features to use for training.
        k (int): Number of folds for Stratified K-Fold Cross-Validation.

    Returns:
        pd.DataFrame: DataFrame containing global LIME feature importances.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    X = df[selected_features].values
    y = df[target_column].values

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Perform Stratified K-Fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the AdaBoost model for the 3rd fold
        if fold == 3:
            model = AdaBoostClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save and reload the model to mimic the process
            model_filename = f"adaboost_fold_{fold}.joblib"
            joblib.dump(model, model_filename)
            loaded_model = joblib.load(model_filename)

            # Initialize LIME Explainer
            explainer = LimeTabularExplainer(
                X_test,
                feature_names=selected_features,
                class_names=[str(cls) for cls in np.unique(y)],
                mode="classification",
                discretize_continuous=True,
            )

            # Compute LIME explanations for multiple samples
            global_importances = {feature: 0 for feature in selected_features}  # Initialize global importances
            num_samples = 20  # Number of samples to explain

            for i in range(min(num_samples, len(X_test))):
                explanation = explainer.explain_instance(
                    X_test[i], loaded_model.predict_proba, num_features=len(selected_features)
                )

                # Process local importances
                for feature, importance in explanation.as_list():
                    # Extract base feature name (e.g., "hemo" from "10.22 < hemo <= 12.80")
                    match = re.search(r'([a-zA-Z_]+)', feature)
                    if match:
                        base_feature = match.group(1)
                        if base_feature in global_importances:
                            global_importances[base_feature] += importance

            # Normalize global importances
            for feature in global_importances:
                global_importances[feature] /= num_samples

            # Create a DataFrame for LIME feature importances
            lime_importance_df = pd.DataFrame({
                'Feature': list(global_importances.keys()),
                'Importance': list(global_importances.values())
            }).sort_values(by='Importance', ascending=False)

            # Plot the LIME feature importances
            plt.figure(figsize=(8, 6))
            plt.barh(lime_importance_df['Feature'], lime_importance_df['Importance'], color='skyblue')
            plt.title("Global LIME Feature Importances (AdaBoost Fold 3)")
            plt.xlabel("LIME Importance")
            plt.ylabel("Feature")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("lime_importances_adaboost_fold_3.png")
            plt.show()

            print("Global LIME feature importances plot saved as 'lime_importances_adaboost_fold_3.png'.")
            return lime_importance_df

# Example Usage
file_path = 'shuffled_dataset.csv'
target_column = 'class'
selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
lime_importances = calculate_lime_feature_importances(file_path, target_column, selected_features, k=10)
print(lime_importances)
