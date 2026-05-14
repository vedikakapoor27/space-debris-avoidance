import joblib
from risk_scorer import score_from_features


def suggest_avoidance(risk, distance_km, rel_velocity):
    """
    Takes risk assessment
    Returns what action to take
    """

    if risk['level'] == 'HIGH':
        # calculate a simple fuel estimate
        fuel_estimate = round(rel_velocity * 0.35, 2)
        maneuver_km   = round(distance_km * 0.1 + 5, 2)

        return {
            'action':         'MANEUVER REQUIRED',
            'maneuver_type':  'Raise orbit',
            'maneuver_km':    f"{maneuver_km} km",
            'fuel_cost_kg':   f"{fuel_estimate} kg",
            'time_window':    'Act within 1 hour',
            'urgency':        'IMMEDIATE'
        }

    elif risk['level'] == 'MEDIUM':
        return {
            'action':         'MONITOR CLOSELY',
            'maneuver_type':  'Prepare orbit adjustment',
            'maneuver_km':    'TBD based on next update',
            'fuel_cost_kg':   'TBD',
            'time_window':    'Review in 2 hours',
            'urgency':        'WATCH'
        }

    else:
        return {
            'action':         'NO ACTION NEEDED',
            'maneuver_type':  'None',
            'maneuver_km':    '0 km',
            'fuel_cost_kg':   '0 kg',
            'time_window':    'Next check in 24 hours',
            'urgency':        'CLEAR'
        }


def full_assessment(model, distance_km, rel_velocity, approach_rate):
    """
    Full pipeline:
    features → risk score → avoidance decision
    """
    # step 1 — get risk
    risk = score_from_features(model, distance_km, rel_velocity, approach_rate)

    # step 2 — get action
    action = suggest_avoidance(risk, distance_km, rel_velocity)

    print(f"\n--- Avoidance Decision ---")
    print(f"  Action      : {action['action']}")
    print(f"  Maneuver    : {action['maneuver_type']}")
    print(f"  Adjust by   : {action['maneuver_km']}")
    print(f"  Fuel needed : {action['fuel_cost_kg']}")
    print(f"  Time window : {action['time_window']}")
    print(f"  Urgency     : {action['urgency']}")

    return {**risk, **action}


# --- test it ---
if __name__ == "__main__":

    model = joblib.load('collision_model.pkl')

    print("=" * 45)
    print("TEST 1 — CRITICAL SCENARIO")
    print("=" * 45)
    full_assessment(model,
                    distance_km=8,
                    rel_velocity=13,
                    approach_rate=-18)

    print("\n" + "=" * 45)
    print("TEST 2 — MEDIUM SCENARIO")
    print("=" * 45)
    full_assessment(model,
                    distance_km=35,
                    rel_velocity=7,
                    approach_rate=-4)

    print("\n" + "=" * 45)
    print("TEST 3 — SAFE SCENARIO")
    print("=" * 45)
    full_assessment(model,
                    distance_km=9000,
                    rel_velocity=2,
                    approach_rate=3)