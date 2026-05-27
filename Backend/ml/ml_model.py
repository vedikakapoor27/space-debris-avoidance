import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data():
    # load Sanidhya's real conjunction data
    real_df = pd.read_csv('conjunction_summary.csv')
    
    # rename columns to match our format
    real_df = real_df.rename(columns={
        'Object1':            'sat1',
        'Object2':            'sat2',
        'Minimum_Distance_km':'distance_km',
        'CPA_Time_sec':       'rel_velocity'  
    })
    
    # add approach rate column
    real_df['approach_rate'] = -1.5
    
    # label from Sanidhya's status
    real_df['collision'] = (real_df['Status'] == 'WARNING').astype(int)
    real_df = real_df[['sat1','sat2','distance_km','rel_velocity','approach_rate','collision']]
    
    # combine with existing simulation data
    sim_df = pd.read_csv('simulation_output.csv')
    
    combined = pd.concat([sim_df, real_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = combined[['distance_km', 'rel_velocity', 'approach_rate']]
    y = combined['collision']
    
    print(f"Real data rows  : {len(real_df)}")
    print(f"Sim data rows   : {len(sim_df)}")
    print(f"Total samples   : {len(combined)}")
    print(f"Collision cases : {y.sum()}")
    return X, y

def train_model(X, y):
    # split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # create the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # train it
    model.fit(X_train, y_train)
    print("Model trained!")
    
    # test it
    y_pred = model.predict(X_test)
    
    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test


def save_model(model):
    joblib.dump(model, 'collision_model.pkl')
    print("\nModel saved as collision_model.pkl")


def predict_single(model, distance_km, rel_velocity, approach_rate):
    """
    Test the model with one example.
    You can call this with any numbers to see what it predicts.
    """
    sample = pd.DataFrame([{
        'distance_km':   distance_km,
        'rel_velocity':  rel_velocity,
        'approach_rate': approach_rate
    }])
    
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    
    result = "DANGER - COLLISION RISK" if prediction == 1 else "SAFE"
    print(f"\nTest prediction:")
    print(f"  distance={distance_km}km, velocity={rel_velocity}, approach={approach_rate}")
    print(f"  Result     : {result}")
    print(f"  Probability: {round(probability * 100, 2)}%")
    return prediction, probability


# --- run everything ---
X, y = load_data()
model, X_test, y_test = train_model(X, y)
save_model(model)

# test with a dangerous scenario
predict_single(model, distance_km=10, rel_velocity=12, approach_rate=-15)

# test with a safe scenario  
predict_single(model, distance_km=8000, rel_velocity=2, approach_rate=1)