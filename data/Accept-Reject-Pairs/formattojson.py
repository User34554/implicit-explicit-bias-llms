import pandas as pd
import json
from sklearn.model_selection import train_test_split

# ==== CONFIG ====
excel_file = "acceptreject_en.xlsx"
output_train_file = "bias_en_train.json"
output_test_file = "bias_en_test.json"
test_fraction = 0.2  # 20% for testing

# ==== LOAD EXCEL ====
df = pd.read_excel(excel_file)

# ==== CREATE DPO FORMAT ====
dpo_data = []

for idx, row in df.iterrows():
    label = row['label']
    sentence = row['sentence']
    response = row['response']

    if label == 1:
        chosen = response    # unbiased
        rejected = sentence  # biased
    elif label == 0:
        chosen = sentence    # unbiased
        rejected = response  # biased
    elif label == 0.5:
        continue
    else:
        raise ValueError(f"Unexpected label {label} at row {idx}")

    dpo_data.append({
        "prompt": sentence,
        "chosen": chosen,
        "rejected": rejected
    })

# ==== SPLIT INTO TRAIN/TEST ====
train_data, test_data = train_test_split(
    dpo_data,
    test_size=test_fraction,
    random_state=42,   # ensures reproducibility
    shuffle=True       # randomize before splitting
)

# ==== SAVE FILES ====
with open(output_train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(output_test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"✅ Conversion complete! Train: {len(train_data)} samples → {output_train_file}, Test: {len(test_data)} samples → {output_test_file}")
