from collections import Counter
from crbac.lidds_detector import load_csv

rows, X, names = load_csv("results/debug_lidds.csv")

print("Total windows:", len(rows))
print("Features:", len(names))
print("Available row fields:", list(rows[0].keys()))

for field in ("split", "partition", "label", "class", "recording_id"):
    if field in rows[0]:
        print(f"\n{field}:")
        print(Counter(str(r[field]) for r in rows))

