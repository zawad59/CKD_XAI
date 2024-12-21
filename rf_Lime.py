import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest_test_cases_with_lime(file_path: str, target_column: str, k: int):
    """
    Random Forest test case predictions with LIME explanations.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
    X = df[selected_features]
    y = df[target_column]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Track the best model
    best_model = None
    best_fold = -1
    best_accuracy = 0

    # Perform Stratified K-Fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Random Forest model
        model = RandomForestClassifier(criterion='gini', random_state=42, n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, model.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_fold = fold

    print(f"Best fold: {best_fold} with accuracy: {best_accuracy:.4f}")

    # Select two test cases: one for CKD (class 0) and one for non-CKD (class 1)
    test_cases = pd.concat([
        X[y == 0].sample(1, random_state=42),  # CKD (class 0)
        X[y == 1].sample(1, random_state=42)  # Non-CKD (class 1)
    ])
    test_case_labels = y[test_cases.index]

    print(f"Test Cases:\n{test_cases}")
    print(f"Actual Labels: {test_case_labels.values}")

    # Apply LIME for explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=selected_features,
        class_names=['CKD (0)', 'Non-CKD (1)'],
        mode='classification',
        random_state=42
    )

    for i, instance in enumerate(test_cases.values):
        label = test_case_labels.iloc[i]
        prediction = best_model.predict([instance])[0]
        prediction_proba = best_model.predict_proba([instance])

        print(f"Test Case {i + 1} (Actual: {'CKD' if label == 0 else 'Non-CKD'}, Predicted: {'CKD' if prediction == 0 else 'Non-CKD'})")
        print(f"Prediction Probabilities: {prediction_proba}")

        explanation = explainer.explain_instance(instance, best_model.predict_proba, num_features=len(selected_features))

        # Plot LIME explanation
        plt.figure(figsize=(8, 6))
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Test Case {i + 1} ({'CKD' if prediction == 0 else 'Non-CKD'})")
        plt.xlabel("Feature Contribution to Prediction")
        plt.ylabel("Feature Value Ranges")
        plt.tight_layout()
        plt.savefig(f"lime_explanation_test_case_{i + 1}.png")
        plt.show()

# File path, target column, and number of folds
file_path = 'shuffled_dataset.csv'
target_column = 'class'
k = 10

# Call the function
random_forest_test_cases_with_lime(file_path, target_column, k)
