import pandas as pd

def prepare_features(df):
    """
    Takes raw dataframe
    Returns X (inputs) and y (labels) ready for ML
    """
    # drop rows with missing values
    df = df.dropna()

    # input columns — what model learns from
    feature_cols = ['distance_km', 'rel_velocity', 'approach_rate']

    # output column — what model predicts
    label_col = 'collision'

    X = df[feature_cols]
    y = df[label_col]

    print(f"Features shape : {X.shape}")
    print(f"Collision cases: {y.sum()}")
    print(f"Safe cases     : {len(y) - y.sum()}")

    return X, y


def get_feature_stats(df):
    """
    Shows basic stats about your data
    Useful to understand what you're working with
    """
    print("\n--- Feature Statistics ---")
    print(df[['distance_km', 'rel_velocity', 'approach_rate']].describe())


# --- test it ---
if __name__ == "__main__":
    df = pd.read_csv('simulation_output.csv')
    get_feature_stats(df)
    X, y = prepare_features(df)
    print("\nFirst 3 rows of X:")
    print(X.head(3))