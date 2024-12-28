import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dice_ml
import seaborn as sns


def generate_predictions_and_counterfactuals(file_path: str, target_column: str):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Define the features used for model training
    selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
    X = df[selected_features]
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    # Combine test data, true labels, and predicted labels for analysis
    test_results = X_test.copy()
    test_results[target_column] = y_test.values
    test_results['predicted_class'] = y_pred
    test_results['predicted_prob_class_1'] = y_pred_prob[:, 1]

    # Filter instances predicted as Class 1 (Non-CKD)
    predicted_class_1 = test_results[test_results['predicted_class'] == 1]
    if predicted_class_1.empty:
        print("\nNo instances predicted as Class 1 (Non-CKD) in the test set.")
        return

    # Select the first instance from the filtered results
    query_instance_index = predicted_class_1.index[0]
    query_instance = test_results.loc[[query_instance_index], selected_features]  # Use only selected features

    print("\nQuery Instance (original prediction: Class 1 - Non-CKD):")
    print(query_instance)

    # Set up Dice for counterfactual generation
    dice_data = df[selected_features + [target_column]]
    data = dice_ml.Data(dataframe=dice_data, continuous_features=selected_features, outcome_name=target_column)
    model_dice = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(data, model_dice)

    # Generate counterfactuals for the desired outcome (Class 0 - CKD)
    explanation = exp.generate_counterfactuals(
        query_instance, total_CFs=5, desired_class=0, features_to_vary=selected_features
    )
    cf_df = explanation.visualize_as_dataframe()

    # Check if counterfactuals are generated
    if cf_df is None or cf_df.empty:
        print("\nNo valid counterfactuals generated for the given query instance.")
        return

    # Display and save counterfactuals as a CSV
    print("\nCounterfactuals (desired outcome: Class 0 - CKD):")
    print(cf_df)
    cf_df.to_csv("counterfactuals_class_1_to_0.csv", index=False)
    print("\nCounterfactuals saved to 'counterfactuals_class_1_to_0.csv'.")

    # Visualize the counterfactuals as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(cf_df[selected_features].astype(float), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Counterfactuals Heatmap (Non-CKD -> CKD)")
    plt.xlabel("Features")
    plt.ylabel("Counterfactual Instances")
    plt.tight_layout()
    plt.savefig("counterfactuals_heatmap.png")
    plt.show()

    print("\nCounterfactuals visualization saved as 'counterfactuals_heatmap.png'.")


# File path and target column
file_path = 'shuffled_dataset.csv'
target_column = 'class'

# Call the function
generate_predictions_and_counterfactuals(file_path, target_column)
