import csv
import json
from pathlib import Path

csv_path = Path('submission/ranked_output.csv')
jsonl_path = Path('cache/candidate_features.jsonl')
out_path = Path('submission/selected_features.jsonl')

if not csv_path.exists() or not jsonl_path.exists():
    print("Files not found.")
    exit(1)

# Get the top 100 CIDs
cids = set()
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cids.add(row['candidate_id'])

# Write only those to selected_features.jsonl
count = 0
with open(jsonl_path, 'r', encoding='utf-8') as fin, \
     open(out_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip(): continue
        try:
            data = json.loads(line)
            if data.get('candidate_id') in cids:
                fout.write(line)
                count += 1
        except Exception:
            pass

print(f"Wrote {count} candidates to {out_path}.")
