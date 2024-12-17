import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def apply_smote(file_path: str, target_column: str, output_file: str):
    """
    Convert categorical features to numeric, then apply SMOTE to handle class imbalance.

    Args:
        file_path (str): Path to the dataset CSV file.
        target_column (str): The target column with class labels.
        output_file (str): Path to save the resampled dataset.

    Returns:
        None: Saves the resampled dataset to the specified output file.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert categorical features to numeric using Label Encoding
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:  # Skip the target column
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target column if it's categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Print class distribution before SMOTE
    print("Class distribution before SMOTE:")
    print(Counter(y))

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Print class distribution after SMOTE
    print("Class distribution after SMOTE:")
    print(Counter(y_resampled))

    # Save the resampled dataset
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled
    resampled_df.to_csv(output_file, index=False)
    print(f"Resampled dataset saved to {output_file}")

# File path, target column, and output file declaration
file_path = 'final_cleaned_file.csv'
target_column = 'class'
output_file = 'smote_resampled_file.csv'

# Call the function
apply_smote(file_path, target_column, output_file)
