from sgp4.api import Satrec, jday
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def parse_tle_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    
    satellites = []
    for i in range(0, len(lines) - 2, 3):
        name  = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        satellites.append((name, line1, line2))
    
    print(f"Loaded {len(satellites)} objects from {filepath}")
    return satellites


def compute_simulation_data(group_a, group_b):
    rows = []
    now = datetime.utcnow()

    for sat_a in group_a[:30]:
        for sat_b in group_b[:30]:

            name_a, l1_a, l2_a = sat_a
            name_b, l1_b, l2_b = sat_b

            obj_a = Satrec.twoline2rv(l1_a, l2_a)
            obj_b = Satrec.twoline2rv(l1_b, l2_b)

            for step in range(5):
                t = now + timedelta(minutes=step * 10)
                jd, fr = jday(t.year, t.month, t.day,
                              t.hour, t.minute, t.second)

                e_a, pos_a, vel_a = obj_a.sgp4(jd, fr)
                e_b, pos_b, vel_b = obj_b.sgp4(jd, fr)

                if e_a != 0 or e_b != 0:
                    continue

                dx = pos_a[0] - pos_b[0]
                dy = pos_a[1] - pos_b[1]
                dz = pos_a[2] - pos_b[2]
                distance_km = (dx**2 + dy**2 + dz**2) ** 0.5

                dvx = vel_a[0] - vel_b[0]
                dvy = vel_a[1] - vel_b[1]
                dvz = vel_a[2] - vel_b[2]
                rel_velocity = (dvx**2 + dvy**2 + dvz**2) ** 0.5

                dot = (dx*dvx + dy*dvy + dz*dvz)
                approach_rate = dot / distance_km

                rows.append({
                    'sat1':          name_a,
                    'sat2':          name_b,
                    'distance_km':   round(distance_km,  3),
                    'rel_velocity':  round(rel_velocity,  3),
                    'approach_rate': round(approach_rate, 3),
                })

    df = pd.DataFrame(rows)
    df['collision'] = 0
    return df


def add_synthetic_danger(df):
    np.random.seed(42)
    n_danger = 500

    synthetic = pd.DataFrame({
        'sat1':          ['IRIDIUM 33 DEB'] * n_danger,
        'sat2':          ['COSMOS 2251 DEB'] * n_danger,
        'distance_km':   np.random.uniform(0.5, 50,  n_danger),
        'rel_velocity':  np.random.uniform(5,   15,  n_danger),
        'approach_rate': np.random.uniform(-20, -5,  n_danger),
        'collision':     [1] * n_danger
    })

    combined = pd.concat([df, synthetic], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


# --- run everything ---
iridium = parse_tle_file('iridium_debris.tle')
cosmos  = parse_tle_file('cosmos_debris.tle')

df = compute_simulation_data(iridium, cosmos)
df = add_synthetic_danger(df)

df.to_csv('simulation_output.csv', index=False)

print(f"\nTotal rows     : {len(df)}")
print(f"Collision cases: {df['collision'].sum()}")
print(f"Safe cases     : {len(df) - df['collision'].sum()}")
print("\nFirst 5 rows:")
print(df.head())