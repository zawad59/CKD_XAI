import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib

def calculate_interpretability_and_fidelity_with_lime(file_path: str, target_column: str, selected_features: list, k: int = 10):
    """
    Calculate Interpretability and Fidelity Scores for the Random Forest Model using LIME.

    Args:
        file_path (str): Path to the dataset.
        target_column (str): Name of the target column.
        selected_features (list): List of features to use for training.
        k (int): Number of folds for Stratified K-Fold Cross-Validation.

    Returns:
        dict: Scores for interpretability, internal fidelity, and external fidelity.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    X = df[selected_features].values
    y = df[target_column].values

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if fold == 3:
            # Train the Random Forest model
            model = RandomForestClassifier(criterion='gini', random_state=42, n_estimators=100, max_depth=5)
            model.fit(X_train, y_train)

            # Save and reload the model
            model_filename = f"random_forest_fold_{fold}.joblib"
            joblib.dump(model, model_filename)
            loaded_model = joblib.load(model_filename)

            # Make predictions
            y_pred = loaded_model.predict(X_test)

            # Initialize LIME explainer
            explainer = LimeTabularExplainer(X_train, feature_names=selected_features, class_names=np.unique(y).astype(str),
                                             discretize_continuous=True, random_state=42)

            # Explain a sample instance from the test set (e.g., the first instance)
            sample_index = 0
            explanation = explainer.explain_instance(X_test[sample_index], loaded_model.predict_proba, num_features=len(selected_features))

            # Extract LIME feature importances
            lime_importance = {feature: weight for feature, weight in explanation.as_list()}

            # Create DataFrame for LIME feature importance
            lime_importance_df = pd.DataFrame({
                'Feature': list(lime_importance.keys()),
                'LIME Importance': list(lime_importance.values())
            }).sort_values(by='LIME Importance', ascending=False)

            # Internal Fidelity: Compare LIME importances with Random Forest feature importances
            rf_importances = loaded_model.feature_importances_
            rf_importance_df = pd.DataFrame({
                'Feature': selected_features,
                'RF Importance': rf_importances
            })
            merged_importances = lime_importance_df.merge(rf_importance_df, on='Feature', how='inner')

            internal_fidelity = np.corrcoef(merged_importances['LIME Importance'], merged_importances['RF Importance'])[0, 1]

            # External Fidelity: Compare LIME surrogate predictions with original Random Forest predictions
            surrogate_pred_class = np.argmax(explanation.local_pred)  # Convert LIME surrogate probabilities to class
            external_fidelity = accuracy_score([y_pred[sample_index]], [surrogate_pred_class])

            # Results
            results = {
                'Interpretability (LIME Feature Importance)': lime_importance_df.to_dict(orient='records'),
                'Internal Fidelity (Correlation with RF)': internal_fidelity,
                'External Fidelity (LIME vs RF Predictions)': external_fidelity
            }

            return results

# File path, target column, and selected features
file_path = 'shuffled_dataset.csv'
target_column = 'class'
selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']

# Calculate scores with LIME
scores = calculate_interpretability_and_fidelity_with_lime(file_path, target_column, selected_features, k=10)
print("Scores:", scores)
