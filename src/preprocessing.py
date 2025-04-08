import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open("data/sarcasm_data.json", "r") as f: 
    data = json.load(f)

utterances = []
labels = []

for key, value in data.items():
    utterances.append(value["utterance"])
    labels.append(1 if value["sarcasm"] else 0)  

df = pd.DataFrame({"utterance": utterances, "label": labels})

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["utterance"], df["label"], test_size=0.2, random_state=42
)

# Save Train and Test Data to CSV Files
train_df = pd.DataFrame({"utterance": train_texts, "label": train_labels})
test_df = pd.DataFrame({"utterance": test_texts, "label": test_labels})

train_df.to_csv("MUStARD_train.csv", index=False)
test_df.to_csv("MUStARD_test.csv", index=False)


print(f"✅ Preprocessing Done: {len(train_texts)} Train, {len(test_texts)} Test Samples")
