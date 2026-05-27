import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
show_plots = False

# ======================================================
# LOAD DATA
# ======================================================

print("Current Folder:")
print(os.getcwd())

data = pd.read_csv(
    "orbit_data.txt",
    sep=r"\s+",
    engine="python",
    on_bad_lines="skip"
)

print("\nData Preview:")
print(data.head())

# ======================================================
# OBJECT LIST
# ======================================================

objects = [
    "ISS",
    "Spacecraft1",
    "Spacecraft2",
    "Spacecraft3"
]

# ======================================================
# COLLISION THRESHOLD
# ======================================================

threshold = 200  # km

# ======================================================
# STORE RESULTS
# ======================================================

results = []

# ======================================================
# LOOP THROUGH ALL OBJECT PAIRS
# ======================================================

for obj1, obj2 in combinations(objects, 2):

    print("\n================================================")
    print(f"ANALYZING: {obj1} ↔ {obj2}")
    print("================================================")

    # --------------------------------------------------
    # POSITION VECTORS
    # --------------------------------------------------

    x1 = data[f"{obj1}.EarthMJ2000Eq.X"]
    y1 = data[f"{obj1}.EarthMJ2000Eq.Y"]
    z1 = data[f"{obj1}.EarthMJ2000Eq.Z"]

    x2 = data[f"{obj2}.EarthMJ2000Eq.X"]
    y2 = data[f"{obj2}.EarthMJ2000Eq.Y"]
    z2 = data[f"{obj2}.EarthMJ2000Eq.Z"]

    # --------------------------------------------------
    # RELATIVE POSITION
    # --------------------------------------------------

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # --------------------------------------------------
    # DISTANCE CALCULATION
    # --------------------------------------------------

    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    # --------------------------------------------------
    # CPA (CLOSEST POINT OF APPROACH)
    # --------------------------------------------------

    cpa_index = np.argmin(distance)

    cpa_time = data["ISS.ElapsedSecs"][cpa_index]

    min_distance = np.min(distance)

    print(f"\nCPA Time: {cpa_time:.2f} seconds")
    print(f"Minimum Distance: {min_distance:.2f} km")

    # --------------------------------------------------
    # COLLISION CHECK
    # --------------------------------------------------

    if min_distance < threshold:

        print("⚠️ COLLISION RISK DETECTED")

        collision_status = "WARNING"

    else:

        print("✅ SAFE")

        collision_status = "SAFE"

    # --------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------

    results.append({
        "Object1": obj1,
        "Object2": obj2,
        "CPA_Time_sec": cpa_time,
        "Minimum_Distance_km": min_distance,
        "Status": collision_status
    })

    # --------------------------------------------------
    # PLOT DISTANCE VS TIME
    # --------------------------------------------------

    time = data["ISS.ElapsedSecs"]

    plt.figure(figsize=(10,6))

    plt.plot(time, distance)

    plt.scatter(cpa_time, min_distance)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Distance (km)")

    plt.title(f"{obj1} vs {obj2}")

    plt.grid(True)

    if show_plots:
      plt.show()

# ======================================================
# FINAL SUMMARY TABLE
# ======================================================

summary = pd.DataFrame(results)

print("\n================================================")
print("FINAL CONJUNCTION ANALYSIS SUMMARY")
print("================================================")

print(summary)

# ======================================================
# EXPORT RESULTS
# ======================================================

summary.to_csv("conjunction_summary.csv", index=False)
# ======================================================
# EXPORT JSON DATASET
# ======================================================

summary.to_json(
    "conjunction_summary.json",
    orient="records",
    indent=4
)

print("\nJSON dataset exported as conjunction_summary.json")

print("\nSummary exported as conjunction_summary.csv")