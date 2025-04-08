import numpy as np
import torch
from sklearn.model_selection import KFold
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load tokenized train and test data
train_data = torch.load("MUStARD_train.pt")
test_data = torch.load("SemEval_test.pt")

# Define seeds from the paper
seeds = [1, 11, 21, 31]

# Create a custom Dataset class
class SarcasmDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = encodings["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Prepare train and test datasets
full_dataset = SarcasmDataset(train_data)
test_dataset = SarcasmDataset(test_data)

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
test_results = []

# Loop through multiple seeds
for seed in seeds:
    print(f"\n🔹 Training with Seed {seed} 🔹")

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"📌 Training Fold {fold+1}/5 with Seed {seed}...")

        # Split data into train/validation
        train_split = torch.utils.data.Subset(full_dataset, train_idx)
        val_split = torch.utils.data.Subset(full_dataset, val_idx)

        # Load BERT model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/seed_{seed}_fold_{fold+1}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=64,
            weight_decay=1e-2,
            num_train_epochs=2,
            logging_dir=f"./logs/seed_{seed}_fold_{fold+1}",
            logging_steps=50,
            seed=seed,
        )

        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_split,
            eval_dataset=val_split,
        )

        # Train the model
        trainer.train()

        # Evaluate the model on validation data
        eval_results = trainer.evaluate()
        print(f"📊 Fold {fold+1} Validation Results (Seed {seed}): {eval_results}")
        fold_results.append(eval_results)

        # Evaluate the model on SemEval Test Data
        test_eval_results = trainer.predict(test_dataset)
        test_accuracy = (test_eval_results.predictions.argmax(-1) == test_eval_results.label_ids).mean()
        print(f"📊 Fold {fold+1} Test Accuracy (Seed {seed}): {test_accuracy:.4f}")
        test_results.append(test_accuracy)

# Compute average performance across all seeds and folds (Validation)
avg_fold_results = {key: np.mean([fold[key] for fold in fold_results]) for key in fold_results[0]}
print(f"✅ Average Validation Results Across 5 Folds & 4 Seeds: {avg_fold_results}")

# Compute average test performance across all seeds and folds
avg_test_accuracy = np.mean(test_results)
print(f"✅ Average Test Accuracy Across 5 Folds & 4 Seeds: {avg_test_accuracy:.4f}")

# Save results
np.save("bert_cross_val_results.npy", avg_fold_results)
np.save("bert_test_results.npy", {"test_accuracy": avg_test_accuracy})
print("✅ Cross-validation and test results saved.")
