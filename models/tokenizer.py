import pandas as pd
import torch
from transformers import BertTokenizer

# Load preprocessed data
train_df = pd.read_csv("MUStARD_train.csv")
test_df = pd.read_csv("MUStARD_test.csv")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize training and test data
train_encodings = tokenizer(
    train_df["utterance"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt"
)
test_encodings = tokenizer(
    test_df["utterance"].tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt"
)

# Convert labels to tensor format
train_labels = torch.tensor(train_df["label"].tolist())
test_labels = torch.tensor(test_df["label"].tolist())

# Save tokenized data
torch.save({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_labels}, "MUStARD_train.pt")
torch.save({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": test_labels}, "MUStARD_test.pt")

print(f"✅ Tokenization Complete: {len(train_labels)} Train, {len(test_labels)} Test Samples")
print("✅ Tokenized data saved as 'MUStARD_train.pt' and 'MUStARD_test.pt'.")
