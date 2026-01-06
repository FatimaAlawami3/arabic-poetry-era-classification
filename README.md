# Arabic Poetry Era Classification

## Overview

This project focuses on **automated recognition of historical eras in Arabic poetry** using Natural Language Processing (NLP). The task is formulated as a **12-class text classification problem**, where each poem is assigned to its corresponding historical era.

The project presents a **comparative study** between traditional deep learning models and modern transformer-based architectures under a **unified experimental pipeline**, allowing fair and controlled evaluation.

---

## Problem Statement

Arabic poetry spans many historical periods, each with distinct linguistic and stylistic characteristics. Automatically identifying the era of a poem is challenging due to:

* Rich Arabic morphology and vocabulary
* Stylistic overlap between adjacent historical eras
* Class imbalance in available datasets

This project aims to address these challenges using deep learning and transformer-based NLP models.

---

## Dataset

* **Dataset name:** FannOrFlop
* **Source:** Hugging Face
* **Size:** ~6,000 Arabic poems
* **Classes:** 12 historical eras
* **Input:** Poem verses (text)
* **Target:** Historical era label

To handle class imbalance, **upsampling** was applied to the training set, ensuring all eras were equally represented during training.

---

## Models Implemented

### Traditional Deep Learning Models

These models were trained **from scratch** using word-level representations:

* **TextCNN** – captures local n-gram features using convolutional filters
* **BiLSTM** – models long-range sequential dependencies in poetic text

### Transformer-Based Models

These models were **fine-tuned** using pretrained Arabic language models:

* **AraBERT** – general-purpose Arabic transformer
* **AraPoemBERT** – pretrained specifically on Arabic poetry
* **QARiB** – pretrained on large and diverse Arabic corpora
* **CAMeLBERT-CA** – pretrained on Classical Arabic

---

## Experimental Setup

* Unified preprocessing and training pipeline for all models
* Train/validation split: 80% / 20% (stratified)
* Loss function: Cross-entropy
* Optimizers:

  * Adam (traditional models)
  * AdamW (transformer models)
* Evaluation metrics:

  * Accuracy
  * Macro F1-score
  * Weighted F1-score

---

## Results Summary

Transformer-based models significantly outperformed traditional deep learning models.

* **Best performing model:** CAMeLBERT-CA
* **Accuracy:** ~70%
* **Key observation:** Most misclassifications occur between historically adjacent eras, reflecting genuine stylistic overlap rather than model weakness.

---

## Project Structure

```
├── notebooks/
│   ├── AraBERT.ipynb
│   ├── AraPoemBERT.ipynb
│   ├── CAMeLBERT-ca.ipynb
│   ├── QARiB.ipynb
│   ├── BiLSTM.ipynb
│   └── TextCNN.ipynb
│
├── report/
│   └── Automated_Era_Recognition_Arabic_Poetry.pdf
│
├── README.md
└── requirements.txt
```

---

## Tools and Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas, NumPy
* Jupyter Notebook

---

## Key Learning Outcomes

* Practical comparison between traditional NLP models and transformers
* Handling class imbalance in multi-class text classification
* Fine-tuning pretrained language models for domain-specific tasks
* Error analysis using confusion matrices

---

## Team

This project was completed as a **university group project**.

---

## Future Work

* Incorporating poetic features such as rhyme and meter
* Using larger and more balanced datasets
* Applying explainable AI techniques to improve interpretability
