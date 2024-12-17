import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


def impute_missing_values_with_logreg(training_file: str, testing_file: str, output_file: str):
    """
    Impute missing values in the testing dataset using Regression trained on the training dataset.
    - For categorical targets, use Logistic Regression.
    - For continuous targets, use Linear Regression.

    In the test dataset, only independent features with values are considered for prediction, ignoring features with nulls.

    Args:
        training_file (str): Path to the training dataset (CSV file).
        testing_file (str): Path to the testing dataset (CSV file).
        output_file (str): Path to save the imputed testing dataset (CSV file).
    """
    # Load the datasets
    training_df = pd.read_csv(training_file)
    testing_df = pd.read_csv(testing_file)

    # Replace any unexpected strings (e.g., '\t?') with NaN
    training_df.replace(r'\t\?|\?', np.nan, regex=True, inplace=True)
    testing_df.replace(r'\t\?|\?', np.nan, regex=True, inplace=True)

    # Encode categorical columns using LabelEncoder
    label_encoders = {}
    for column in training_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        training_df[column] = le.fit_transform(training_df[column].astype(str))
        label_encoders[column] = le

    for column in testing_df.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            le = label_encoders[column]
            testing_df[column] = testing_df[column].apply(
                lambda x: le.transform([x])[0] if pd.notna(x) and x in le.classes_ else -1)

    # Iterate over columns with missing values in the testing dataset
    for target_column in testing_df.columns:
        if testing_df[target_column].isnull().any():
            print(f"Imputing missing values for column: {target_column}")

            # Identify valid independent features in the test set
            valid_features = testing_df.drop(columns=[target_column]).notnull().all(axis=0)
            valid_features = valid_features[valid_features].index.tolist()

            # Filter training data to match valid features
            X_train = training_df[valid_features].dropna()
            y_train = training_df.loc[X_train.index, target_column].dropna()

            # Ensure there is sufficient data to train the model
            if y_train.empty or X_train.empty:
                print(f"Skipping {target_column} due to lack of training data.")
                continue

            # Determine if the target is continuous or categorical
            if training_df[target_column].dtype in ['float64', 'int64']:
                model = LinearRegression()
            else:
                model = LogisticRegression(max_iter=2000)

            # Train the model
            model.fit(X_train, y_train)

            # Impute missing values in the test set
            missing_indices = testing_df[target_column].isnull()
            for idx in testing_df[missing_indices].index:
                row = testing_df.loc[idx, valid_features].dropna()
                if not row.empty:
                    row_input = row.values.reshape(1, -1)
                    prediction = model.predict(row_input)
                    testing_df.at[idx, target_column] = prediction
                else:
                    print(f"Skipping row {idx} for column {target_column} due to lack of available features.")

    # Decode categorical columns back to original labels
    for column, le in label_encoders.items():
        if column in testing_df.columns:
            testing_df[column] = testing_df[column].apply(
                lambda x: le.inverse_transform([int(x)])[0] if pd.notna(x) and x != -1 else x)

    # Save the imputed dataset
    testing_df.to_csv(output_file, index=False)
    print(f"Imputed testing dataset saved to {output_file}")



training_file = 'ckd_complete_rows.csv'
testing_file = 'ckd_unique_missing.csv'
output_file = 'imputed_testing_file.csv'

# Call the function
impute_missing_values_with_logreg(training_file, testing_file, output_file)



