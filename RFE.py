import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def recursive_feature_elimination(file_path: str, target_feature: str, n_features: int):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Encode categorical features into numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Split into features and target
    X = df.drop(columns=[target_feature])
    y = df[target_feature]

    # Perform RFE using Logistic Regression
    model = LogisticRegression(max_iter=2000)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)

    # Get the list of selected features
    selected_features = X.columns[rfe.support_].tolist()

    # Print and return the selected features
    print(f"Selected Features ({n_features}) using RFE:\n")
    print(selected_features)
    return selected_features

# File path, target feature, and number of features declaration
file_path = 'final_cleaned_file.csv'
target_feature = 'class'
n_features = 10

# Call the function
selected_features = recursive_feature_elimination(file_path, target_feature, n_features)
