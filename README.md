# Multimodal Sarcasm Detection (Thesis Project)

This project investigates the generalizability of sarcasm detection models across datasets using multimodal features from text, visual, and (eventually) audio sources. It is part of my Master’s thesis at Colorado State University.

## 🔍 Overview
- Goal: Evaluate how well sarcasm detection models trained on one dataset (MUStARD++) perform on another (SemEval).
- Modalities: Text (RoBERTa-based), Visual (OpenFace), Audio (planned).
- Focus: Understanding modality importance and transferability in a resource-efficient setup.

## 📁 Datasets
- **MUStARD++**: Used for training and evaluation.
- **SemEval**: Used as a transfer domain for generalization testing.

## 🧠 Current Progress
- Text-based sarcasm classification using **RoBERTa** (HuggingFace Transformers).
- Visual feature extraction using **OpenFace** — working for single videos; batch processing in progress.
- Planning audio feature integration as next step (targeting tools like OpenSMILE or pyAudioAnalysis).

## 🧰 Tools & Libraries
- Python, PyTorch, HuggingFace Transformers
- OpenFace (visual)
- scikit-learn, pandas, NumPy, matplotlib

## 📌 Next Steps
- Complete visual feature batch extraction pipeline
- Integrate audio modality
- Cross-dataset evaluation + final reporting
