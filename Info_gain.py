import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy

def calculate_entropy(series):
    """Calculate entropy of a feature."""
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts, base=2)

def calculate_information_gain_with_entropy(file_path: str, target_feature: str):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Encode categorical features, including target variable
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Define features and target
    X = df.drop(columns=[target_feature])
    y = df[target_feature]

    # Calculate entropy for each feature and target
    entropies = {col: calculate_entropy(df[col]) for col in X.columns}
    target_entropy = calculate_entropy(y)

    # Calculate information gain
    info_gain = mutual_info_classif(X, y, discrete_features='auto')
    info_gain_df = pd.DataFrame({
        'Feature': X.columns,
        'Entropy': [entropies[col] for col in X.columns],
        'Information Gain': info_gain
    }).sort_values(by='Information Gain', ascending=False)

    # Print results
    print(f"Target Entropy: {target_entropy}\n")
    print("Feature Entropy and Information Gain Values:")
    print(info_gain_df)

    # Save the results to a CSV
    output_file = "information_gain_with_entropy_results.csv"
    info_gain_df.to_csv(output_file, index=False)
    print(f"Information gain and entropy results saved to {output_file}")

# File path and target feature declaration
file_path = 'final_cleaned_file.csv'
target_feature = 'class'

# Call the function
calculate_information_gain_with_entropy(file_path, target_feature)
