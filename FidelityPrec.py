import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold

def visualize_feature_importances_from_tree(file_path: str, target_column: str, selected_features: list, k: int = 10):
    """
    Visualize Feature Importances and Decision Tree splits using Gini Index for the best fold in K-Fold Cross-Validation.

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

    # Perform Stratified K-Fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the Decision Tree model
        model = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)
        model.fit(X_train, y_train)

        # Evaluate model
        accuracy = model.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_fold = fold
            X_test_best = X_test

    print(f"Best Fold: {best_fold} with Accuracy: {best_accuracy:.4f}")

    # Identify Feature Importances
    feature_importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature Importances (based on Gini Index):")
    print(feature_importance_df)

    # Plot Feature Importances
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.title("Global Feature Importances (Decision Tree - Best Fold)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("decision_tree_feature_importances.png")
    plt.show()

    # Visualize the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(best_model, feature_names=selected_features, class_names=[str(cls) for cls in np.unique(y)],
              filled=True, rounded=True)
    plt.title(f"Decision Tree Visualization (Best Fold: {best_fold})")
    plt.savefig("decision_tree_visualization.png")
    plt.show()

# Example Usage
file_path = 'shuffled_dataset.csv'
target_column = 'class'
selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
visualize_feature_importances_from_tree(file_path, target_column, selected_features, k=10)
