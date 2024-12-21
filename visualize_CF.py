import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_counterfactuals(query_instance, cf_df, features, save_path="counterfactuals_visualization.png"):
    """
    Visualizes the query instance and counterfactuals using a heatmap.

    Args:
    - query_instance (pd.DataFrame): The original query instance as a DataFrame.
    - cf_df (pd.DataFrame): The generated counterfactuals as a DataFrame.
    - features (list): List of features to visualize.
    - save_path (str): File path to save the image.
    """
    # Combine query instance and counterfactuals
    query_instance["type"] = "Original"
    cf_df["type"] = ["Counterfactual " + str(i + 1) for i in range(len(cf_df))]
    combined_df = pd.concat([query_instance, cf_df], ignore_index=True)

    # Filter only the relevant features
    combined_df = combined_df[features + ["type"]]

    # Convert to long format for visualization
    melted_df = combined_df.melt(id_vars=["type"], var_name="Feature", value_name="Value")

    # Ensure pivot table doesn't have duplicates
    pivot_df = melted_df.pivot_table(index="type", columns="Feature", values="Value", aggfunc="first")

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        linewidths=0.5
    )
    plt.title("Query Instance and Counterfactuals Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Instance Type")
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(save_path)
    plt.show()
    print(f"Counterfactual visualization saved as {save_path}")


# Assuming query_instance and cf_df are available
# Query Instance
query_instance = pd.DataFrame({
    "hemo": [15.145378],
    "al": [0.0],
    "htn": [1],
    "dm": [1],
    "sc": [1.0],
    "age": [24.0],
    "class": [1]
})

# Counterfactuals DataFrame
cf_df = pd.DataFrame({
    "hemo": [15.145378, 4.900000, 15.145378, 15.145378, 16.900000],
    "al": [2.9, 1.5, 0.0, 1.5, 1.7],
    "htn": [1, 1, 1, 1, 1],
    "dm": [1, 1, 1, 1, 0],
    "sc": [34.9, 1.0, 41.1, 1.0, 1.0],
    "age": [24.0, 24.0, 24.0, 24.0, 24.0],
    "class": [0, 0, 0, 0, 0]
})

# Features for visualization
features = ["hemo", "al", "htn", "dm", "sc", "age"]

# Generate visualization
visualize_counterfactuals(query_instance, cf_df, features)
