import pandas as pd
import json
import os

# ==== CONFIG ====
excel_file = "acceptreject_de.xlsx"  # path to your Excel file
output_train_file = "bias_de_train.json"  # output JSON file
output_test_file = "bias_de_test.json"  # optional, you can split later
test_fraction = 0.1  # fraction of data to use for testing

# ==== LOAD EXCEL ====
df = pd.read_excel(excel_file)

# ==== CREATE DPO FORMAT ====
dpo_data = []

for idx, row in df.iterrows():
    label = row['label']
    sentence = row['sentence']
    response = row['response']

    # Map to chosen/rejected based on label
    if label == 1:
        # sentence is biased, response is unbiased
        chosen = response  # unbiased
        rejected = sentence  # biased
    elif label == 0:
        # sentence is unbiased, response is biased
        chosen = sentence  # unbiased
        rejected = response  # biased
    elif label == 0.5:
        continue
    else:
        raise ValueError(f"Unexpected label {label} at row {idx}")

    # Add prompt (optional: can just reuse sentence)
    print(f"Processing row {idx}:")
    print(f"  Prompt: {sentence}")
    print(f"  Chosen (unbiased): {chosen}")
    print(f"  Rejected (biased): {rejected}")

    dpo_data.append({
        "prompt": sentence,
        "chosen": chosen,
        "rejected": rejected
    })
    # ==== OPTIONAL: SPLIT INTO TRAIN/TEST ====
    if test_fraction > 0:
        test_size = int(len(dpo_data) * test_fraction)
        train_data = dpo_data[:-test_size]
        test_data = dpo_data[-test_size:]

        with open(output_train_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(output_test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
    # save all as train if no test split
    with open(output_train_file, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

print(f"✅ Conversion complete! Train: {output_train_file}, Test: {output_test_file if test_fraction > 0 else 'No testset'}")
