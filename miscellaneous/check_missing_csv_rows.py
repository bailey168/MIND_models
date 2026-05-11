import pandas as pd

with open('/Users/baileyng/MIND_models/aparc_base_id.txt', 'r') as f:
    aparc_ids = set()
    for line in f:
        line = line.strip()
        if line and not line.startswith('//'):
            aparc_ids.add(line)

df = pd.read_csv('/Users/baileyng/MIND_models/ukb_tabular.csv', dtype=str)
ukb_eids = set(df['eid'].astype(str))

missing_ids = sorted(aparc_ids - ukb_eids)

print(f"IDs in aparc_base_id.txt not in ukb_tabular.csv (eid):")
for eid in missing_ids:
    print(eid)

print(f"\nTotal missing: {len(missing_ids)}")