import pandas as pd
import os

def load_simulation_data():
    filepath = 'simulation_output.csv'

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# --- test it ---
if __name__ == "__main__":
    df = load_simulation_data()
    print(df.head())