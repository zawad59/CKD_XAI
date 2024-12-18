import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def svm_with_kfold_and_grid_search(file_path: str, target_column: str, k: int):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Shuffle the dataset
    #df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    selected_features = ['hemo', 'rbc', 'al', 'htn', 'pot', 'dm']
    # Split into features and target
    #X = df.drop(columns=[target_column])
    X = df[selected_features]
    y = df[target_column]

    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Initialize Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize metric trackers
    all_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    class_metrics = {"precision": [], "recall": [], "f1": [], "class_accuracy": []}

    # Perform Stratified K-Fold cross-validation
    fold = 1
    for train_index, test_index in skf.split(X, y):
        print(f"Fold {fold}:")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Perform GridSearchCV to find the best SVM parameters
        grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get the best SVM model
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for Fold {fold}: {grid_search.best_params_}")

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        overall_precision = precision_score(y_test, y_pred, average='weighted')
        overall_recall = recall_score(y_test, y_pred, average='weighted')
        overall_f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Track metrics
        all_metrics["accuracy"].append(accuracy)
        all_metrics["precision"].append(overall_precision)
        all_metrics["recall"].append(overall_recall)
        all_metrics["f1"].append(overall_f1)

        class_metrics["precision"].append(precision)
        class_metrics["recall"].append(recall)
        class_metrics["f1"].append(f1)
        class_metrics["class_accuracy"].append(class_accuracy)

        # Print metrics for this fold
        print(f"Overall Metrics for Fold {fold}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall: {overall_recall:.4f}")
        print(f"  F1 Score: {overall_f1:.4f}")
        for i, class_label in enumerate(np.unique(y)):
            print(f"Class {class_label}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1 Score: {f1[i]:.4f}")
            print(f"  Accuracy: {class_accuracy[i]:.4f}")
        fold += 1

    # Calculate mean metrics across all folds
    print("\nBest Mean Metrics Across All Folds:")
    print(f"Overall Accuracy: {np.mean(all_metrics['accuracy']):.4f}")
    print(f"Overall Precision: {np.mean(all_metrics['precision']):.4f}")
    print(f"Overall Recall: {np.mean(all_metrics['recall']):.4f}")
    print(f"Overall F1 Score: {np.mean(all_metrics['f1']):.4f}")
    for i, class_label in enumerate(np.unique(y)):
        mean_precision = np.mean([p[i] for p in class_metrics['precision']])
        mean_recall = np.mean([r[i] for r in class_metrics['recall']])
        mean_f1 = np.mean([f[i] for f in class_metrics['f1']])
        mean_class_acc = np.mean([ca[i] for ca in class_metrics['class_accuracy']])
        print(f"Class {class_label}:")
        print(f"  Mean Precision: {mean_precision:.4f}")
        print(f"  Mean Recall: {mean_recall:.4f}")
        print(f"  Mean F1 Score: {mean_f1:.4f}")
        print(f"  Mean Accuracy: {mean_class_acc:.4f}")

# File path, target column, and number of folds
file_path = 'shuffled_dataset.csv'
target_column = 'class'
k = 10

# Call the function
svm_with_kfold_and_grid_search(file_path, target_column, k)
