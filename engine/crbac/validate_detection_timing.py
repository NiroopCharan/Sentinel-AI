import csv
import json
import numpy as np
from collections import defaultdict

with open("results/debug_lidds.csv", encoding="utf-8") as f:
    source = list(csv.DictReader(f))

with open("results/sequential_vs_mad_per_window.csv", encoding="utf-8") as f:
    scores = list(csv.DictReader(f))

metadata = {}
for r in source:
    if r["split"] == "test":
        metadata.setdefault(r["recording_id"], r)

groups = defaultdict(list)
for r in scores:
    groups[r["recording_id"]].append(r)

attack_detected = 0
attack_total = 0
benign_false = 0
benign_total = 0
pre_exploit_alarms = 0
delays = []

for rid, items in groups.items():
    meta = metadata[rid]
    items.sort(key=lambda x: float(x["window_start"]))

    is_attack = (
        meta["category"] == "normal_and_attack"
        and meta["exploit"].strip().lower() == "true"
    )

    alarms = [
        x for x in items
        if float(x["e_value"]) >= 100
    ]

    if is_attack:
        attack_total += 1
        exploit_time = float(meta["exploit_time"])

        pre = [
            x for x in alarms
            if float(x["window_start"]) < exploit_time
        ]
        post = [
            x for x in alarms
            if float(x["window_start"]) >= exploit_time
            and x["label"] == "1"
        ]

        if pre:
            pre_exploit_alarms += 1

        if post:
            attack_detected += 1
            delays.append(
                max(0, float(post[0]["window_start"]) - exploit_time)
            )
    else:
        benign_total += 1
        if alarms:
            benign_false += 1

print("TEMPORAL DETECTION VALIDATION")
print("-----------------------------")
print("Attack recordings:", attack_total)
print("Detected attack recordings:", attack_detected)
print("Attack detection rate:", round(attack_detected / attack_total, 4))
print("Pre-exploit alarm recordings:", pre_exploit_alarms)
print("Benign recordings:", benign_total)
print("Benign recordings with alarms:", benign_false)
print("Benign recording FPR:", round(benign_false / benign_total, 4))
print("Median detection delay:", round(float(np.median(delays)), 4) if delays else None)
print("Detected attack delays:", len(delays))
