import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to apply SHAP on a random test case with RandomForestClassifier
def shap_explainer_train_test_split_rf(file_path: str, target_column: str):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Features and target selection
    selected_features = ['hemo', 'al', 'htn', 'dm', 'sc', 'age']
    X = df[selected_features]
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Debugging: Print SHAP values
    print("SHAP values shape:", shap_values.values.shape)
    print("First test case SHAP values (class 0):", shap_values.values[0, :, 0])

    # Generate SHAP summary plot for Class 0 (CKD)
    shap_values_class_0 = shap_values[..., 0]  # Extract SHAP values for class 0
    shap.summary_plot(shap_values_class_0, X_test, feature_names=selected_features,show=False)
    plt.savefig("shap_summary_class_0_ckd.png", bbox_inches="tight",dpi=700)
    plt.close()  # Close the plot to avoid overlap

    # Generate SHAP summary plot for Class 1 (Non-CKD)
    shap_values_class_1 = shap_values[..., 1]  # Extract SHAP values for class 1
    shap.summary_plot(shap_values_class_1, X_test, feature_names=selected_features,show=False)
    plt.savefig("shap_summary_class_1_non_ckd.png", bbox_inches="tight",dpi=700)
    plt.close()

    # Generate SHAP dependence plot for "hemo" with "al" as interaction
    shap.dependence_plot("hemo", shap_values.values[..., 0], X_test, interaction_index="al",show=False)
    plt.savefig("shap_dependence_plot_hemo.png", bbox_inches="tight",dpi=700)
    plt.close()

# File path and target column
file_path = 'shuffled_dataset.csv'
target_column = 'class'

# Call the function
shap_explainer_train_test_split_rf(file_path, target_column)
