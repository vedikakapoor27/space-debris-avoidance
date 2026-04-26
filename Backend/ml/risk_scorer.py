def get_risk_score(probability):
    """
    Takes collision probability (0 to 1)
    Returns risk level + what it means
    """
    if probability >= 0.7:
        return {
            'level':       'HIGH',
            'color':       'RED',
            'probability': round(probability * 100, 2),
            'message':     'Immediate action required — collision risk is critical'
        }
    elif probability >= 0.3:
        return {
            'level':       'MEDIUM',
            'color':       'ORANGE',
            'probability': round(probability * 100, 2),
            'message':     'Monitor closely — situation developing'
        }
    else:
        return {
            'level':       'LOW',
            'color':       'GREEN',
            'probability': round(probability * 100, 2),
            'message':     'No immediate threat detected'
        }


def score_from_features(model, distance_km, rel_velocity, approach_rate):
    """
    Takes the trained model + satellite data
    Returns full risk assessment
    """
    import pandas as pd

    sample = pd.DataFrame([{
        'distance_km':   distance_km,
        'rel_velocity':  rel_velocity,
        'approach_rate': approach_rate
    }])

    probability = model.predict_proba(sample)[0][1]
    risk = get_risk_score(probability)

    print(f"\n--- Risk Assessment ---")
    print(f"  Distance      : {distance_km} km")
    print(f"  Rel Velocity  : {rel_velocity} km/s")
    print(f"  Approach Rate : {approach_rate}")
    print(f"  Probability   : {risk['probability']}%")
    print(f"  Risk Level    : {risk['level']} ({risk['color']})")
    print(f"  Message       : {risk['message']}")

    return risk


# --- test it ---
if __name__ == "__main__":
    import joblib

    model = joblib.load('collision_model.pkl')

    print("=== TEST 1 - Very dangerous ===")
    score_from_features(model, distance_km=5,    rel_velocity=13, approach_rate=-18)

    print("=== TEST 2 - Medium risk ===")
    score_from_features(model, distance_km=30,   rel_velocity=8,  approach_rate=-4)

    print("=== TEST 3 - Safe ===")
    score_from_features(model, distance_km=9000, rel_velocity=2,  approach_rate=3)