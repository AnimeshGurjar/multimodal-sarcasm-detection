import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification

# Load tokenized test data
test_data = torch.load("MUStARD_test.pt")

# Load trained model
model = BertForSequenceClassification.from_pretrained("./sarcasm_bert_model")
model.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare test data for evaluation
input_ids = test_data["input_ids"].to(device)
attention_mask = test_data["attention_mask"].to(device)
labels = test_data["labels"].to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Convert predictions and labels to numpy
preds = predictions.cpu().numpy()
true_labels = labels.cpu().numpy()

# Compute Metrics
accuracy = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds)
precision = precision_score(true_labels, preds)
recall = recall_score(true_labels, preds)

# Print Results
print(f"✅ Evaluation Complete!")
print(f"📊 Accuracy: {accuracy:.4f}")
print(f"📊 F1-Score: {f1:.4f}")
print(f"📊 Precision: {precision:.4f}")
print(f"📊 Recall: {recall:.4f}")

# Save results to a file
results = {
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall
}
np.save("bert_mustard_results.npy", results)
print("✅ Evaluation results saved to 'bert_mustard_results.npy'.")
