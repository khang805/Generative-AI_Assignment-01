# Generative AI – Assignment #1

**Department of Computer Science**
**National University of Computer and Emerging Sciences (FAST-NUCES)**
**Islamabad, Pakistan**

**Submission Date:** September 21, 2025
**Submitted by:** *Muhammad Abdurrahman Khan (i221148)*

---

## 1. Overview

This repository presents the implementation and evaluation of three generative deep learning models as part of **Generative AI Assignment #1**. The assignment focuses on understanding and implementing convolutional, recurrent, and autoregressive generative architectures using modern deep learning frameworks (TensorFlow and PyTorch).

The repository is organized into three main components corresponding to the assignment questions:

1. **Convolutional Neural Network (CNN)** for image classification using the CIFAR-10 dataset.
2. **Recurrent Neural Network (RNN)** for next-word prediction using Shakespeare’s text corpus.
3. **PixelCNN, Row LSTM, and Diagonal BiLSTM (PixelRNN)** for autoregressive image generation.

---

## 2. Repository Structure

```
│
├── q1.py                 # Implementation of CNN on CIFAR-10
├── q2.py                 # Implementation of RNN (LSTM) for next-word prediction
├── q3.py                 # Implementation of PixelCNN, RowLSTM, and Diagonal BiLSTM
│
├── report.pdf            # Technical report (Springer LNCS format)
└── README.md             # Documentation file
```

## 3. Question 1 — CNN for CIFAR-10 Image Classification

### Objective

To design, train, and evaluate a Convolutional Neural Network (CNN) model on the **CIFAR-10** dataset for image classification, followed by visualization and ablation analysis.

### Key Methodology

* Dataset loaded directly from Hugging Face (`cifar10`).
* Data preprocessing and normalization applied to 60,000 RGB images (32×32×3).
* A 7-layer CNN architecture was developed with **Batch Normalization** and **Dropout** regularization.
* Training conducted using the **Adam optimizer** with a learning rate of 0.001.
* Ablation study performed across hyperparameters: learning rate, batch size, number of filters, and network depth.

### Evaluation

* **Confusion Matrix** and **Classification Metrics** (Accuracy, Precision, Recall, F1-Score) computed using scikit-learn.
* Visualization of **feature maps** extracted from convolutional layers to analyze learned representations.

**Observation:**
The CNN achieved robust generalization on test data. Deeper architectures demonstrated improved feature extraction but required regularization to prevent overfitting.

**Dataset Source:** [Hugging Face – CIFAR-10](https://huggingface.co/datasets/cifar10)

---

## 4. Question 2 — RNN for Next-Word Prediction

### Objective

To develop a Recurrent Neural Network (RNN) capable of generating coherent English text sequences using the **Tiny Shakespeare** dataset.

### Key Methodology

* Text preprocessed and tokenized at the word level (vocabulary size limited to 8,000 words).
* Custom embedding layer trained alongside model parameters.
* Network architecture: **Embedding → BiLSTM → LSTM → Dense Layers**.
* Implemented **temperature-based sampling** for diverse text generation.
* Evaluated using **Perplexity** and **Accuracy** metrics.

**Observation:**
Lower perplexity values and coherent generation indicate strong learning of sequential dependencies. Increasing embedding dimensions and hidden units improved linguistic consistency.

**Dataset Source:** [Hugging Face – Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare)

---

## 5. Question 3 — PixelCNN, Row LSTM, and Diagonal BiLSTM (PixelRNN)

### Objective

To implement and compare three autoregressive image generation architectures—**PixelCNN**, **Row LSTM**, and **Diagonal BiLSTM**—based on the paper *“Pixel Recurrent Neural Networks” (van den Oord et al., 2016)*.

### Key Methodology

* Implemented **masked convolutions** (Type A/B) for PixelCNN.
* Designed **Row LSTM** to process image rows sequentially (top-to-bottom).
* Implemented **Diagonal BiLSTM** with input skewing and bidirectional LSTMs for diagonal context propagation.
* Models trained using **negative log-likelihood (bits per dimension)** as evaluation metric.
* Comparative performance analysis conducted across training and validation losses.

**Observation:**
The Diagonal BiLSTM exhibited superior generative capacity by leveraging both horizontal and vertical pixel dependencies, closely replicating results from the original paper.

**Reference Paper:** [Pixel Recurrent Neural Networks (2016)](https://arxiv.org/abs/1601.06759)

---

## 6. Report Summary

The accompanying **report.pdf** (Springer LNCS format) provides:

* Abstract, Introduction, and Methodology Sections
* Quantitative and qualitative analysis of all models
* Confusion matrices, feature visualizations, and training curves
* Ablation studies and comparative discussion

---

## 7. Author Information

**Name:** Muhammad Abdurrahman Khan
**Roll Number:** i221148
**Program:** BS Computer Science (Session 2022–2026)
**Course:** Generative AI
**Instructor:** Dr. Akhtar Jamil

---

## 8. References

1. Oord, A. van den, Kalchbrenner, N., & Kavukcuoglu, K. (2016). *Pixel Recurrent Neural Networks.* arXiv:1601.06759.
2. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images (CIFAR-10).*
3. Karpathy, A. (2015). *Tiny Shakespeare Dataset.*
4. TensorFlow Documentation – [https://www.tensorflow.org](https://www.tensorflow.org)
5. PyTorch Documentation – [https://pytorch.org](https://pytorch.org)

---
