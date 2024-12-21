import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from alibi.explainers import ALE

# Function to generate ALE plot for the feature 'hemo' in the 3rd fold
def ale_plot_rf(file_path: str, target_column: str):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
    X = df[selected_features]
    y = df[target_column]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform Stratified K-Fold and pick the 3rd fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        if fold == 3:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train RandomForestClassifier model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Initialize ALE explainer
            ale = ALE(model.predict, feature_names=selected_features)
            explanation = ale.explain(X_test.values)

            # Extract data for 'hemo' (index 0)
            hemo_ale_values = explanation.ale_values[0]  # ALE values for 'hemo'
            hemo_feature_values = explanation.feature_values[0]  # Feature bins for 'hemo'

            # Plot ALE for 'hemo'
            plt.figure(figsize=(8, 6))
            plt.plot(hemo_feature_values, hemo_ale_values, marker='o', label='ALE for hemo')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.title("ALE Plot for 'hemo' in CKD Prediction (3rd Fold)")
            plt.xlabel("Hemoglobin Levels")
            plt.ylabel("Contribution Score")
            plt.legend()
            plt.savefig("ale_plot_hemo_fold_3.png", bbox_inches="tight")
            plt.show()

# File path and target column
file_path = 'shuffled_dataset.csv'
target_column = 'class'

# Call the function
ale_plot_rf(file_path, target_column)
