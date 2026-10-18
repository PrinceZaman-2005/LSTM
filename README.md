# ⚡ Long Short-Term Memory (LSTM)

A Long Short-Term Memory (LSTM) network built with PyTorch to explore recurrent computation, gated memory, and sequential learning.

This project demonstrates how LSTMs process text sequences and use gated mechanisms to regulate information flow. The model is trained to classify IMDb movie reviews as positive or negative.

## Introduction

A **Long Short-Term Memory (LSTM)** network is a type of Recurrent Neural Network (RNN) designed to help learn dependencies across sequential data.

Unlike a vanilla RNN, an LSTM maintains both a hidden state and a cell state. Three gates—the forget gate, input gate, and output gate—regulate how information is retained, updated, and exposed.

LSTMs are commonly used for tasks involving text, time series, and other sequential data.

### Architecture

```text
           IMDb Review
                |
                v
             Token IDs
                |
                v
          Embedding Layer
                |
                v
            LSTM Layer
           /          \
     Hidden State   Cell State
           |
           v
    Final Hidden State
           |
           v
       Linear Layer
           |
           v
      Sentiment Logit
           |
           v
    Positive / Negative
```

The embedding layer converts token IDs into vectors. The LSTM processes these vectors sequentially, updating its hidden and cell states at each timestep.

The final hidden state is passed to a linear layer to predict the sentiment of the entire review.

### How Does It Work?

1. **Input sequence:** The model receives tokenized movie reviews.
2. **Embedding:** Each token ID is converted into a trainable vector representation.
3. **Forget gate:** Regulates how much information from the previous cell state is retained.
4. **Input gate:** Regulates how much new information enters the cell state.
5. **Cell-state update:** Combines retained information with selected new information.
6. **Output gate:** Controls how the cell state contributes to the hidden state.
7. **Classification:** The final hidden state is mapped to a sentiment logit.
8. **Learning:** Backpropagation Through Time calculates gradients, allowing the optimizer to update the model parameters.

The objective is to learn useful sequential representations for sentiment classification.

> LSTMs are designed to help preserve information across timesteps, but they do not guarantee that every relevant detail of a sequence will be retained.

## 📊 Dataset

This project uses the **IMDb Large Movie Review Dataset**.

**Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (Hugging Face)

| Property                |                      Value |
| ----------------------- | -------------------------: |
| Total reviews           |                     50,000 |
| Training reviews        |                     25,000 |
| Testing reviews         |                     25,000 |
| Classes                 | Positive (1), Negative (0) |
| Maximum vocabulary size |                     10,000 |
| Maximum review length   |                  80 tokens |

### Data Preprocessing

* Convert reviews to lowercase.
* Tokenize text into words.
* Build a vocabulary using training reviews only.
* Convert words into integer token IDs.
* Replace unknown words with `<UNK>`.
* Pad shorter reviews and truncate longer ones to 80 tokens.
* Use packed sequences to prevent trailing padding from being processed as actual review tokens.

**Special tokens:**

* `<PAD>` → `0`
* `<UNK>` → `1`

## 🏗️ Model Configuration

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Embedding dimension     |                  64 |
| Hidden size             |                  64 |
| Number of LSTM layers   |                   1 |
| Maximum sequence length |                  80 |
| Batch size              |                  32 |
| Optimizer               |                Adam |
| Learning rate           |               0.001 |
| Loss function           | `BCEWithLogitsLoss` |
| Training epochs         |                  10 |

The model uses an embedding layer, a single LSTM layer, and a linear output layer.

The output layer produces one logit per review. During inference, the sigmoid function can convert this logit into a probability.

* Probability ≥ 0.5: Positive
* Probability < 0.5: Negative

During training, raw logits are passed directly to `BCEWithLogitsLoss`.

## 📈 Experimental Results

The model was trained for 10 epochs on a CPU.

| Metric                  |     Result |
| ----------------------- | ---------: |
| Best test accuracy      | **78.88%** |
| Best epoch              |          5 |
| Final training accuracy |     96.04% |
| Final test accuracy     |     78.04% |

### Training Observations

The model's training accuracy increased throughout training, but test performance stopped improving around Epoch 5.

Test loss generally increased during the later epochs, suggesting overfitting.

This experiment demonstrates that improved training accuracy does not necessarily translate into improved generalization.

The best observed test accuracy was achieved before the final epoch, highlighting the importance of evaluating performance throughout training.

*Results represent one training run and may vary with initialization, data order, and other experimental conditions.*

## ⚖️ Trade-Off: Gated Memory vs. Computational Complexity

LSTMs introduce additional mechanisms to regulate information flow compared with vanilla RNNs.

### Advantages

* **Gated information flow:** Controls how existing information is retained and new information is incorporated.
* **Cell-state pathway:** Provides a mechanism for carrying information across timesteps.
* **Sequential processing:** Preserves the order of the input sequence.
* **Parameter sharing:** Reuses recurrent parameters across timesteps.
* **Broad applicability:** Can be used for text classification, time series, and other sequential tasks.

### Trade-Offs

* **Sequential computation:** Recurrent dependencies limit parallel computation across timesteps.
* **Higher computational cost:** Gates introduce additional calculations and parameters compared with vanilla RNNs.
* **Long-term dependency challenges:** Gating helps with information retention but does not guarantee that useful information will always survive.
* **Overfitting:** A model can fit its training data without achieving corresponding improvements on unseen data.
* **Sequence truncation:** Information beyond the configured maximum sequence length is discarded.

LSTMs address some learning difficulties associated with vanilla RNNs, but their effectiveness depends on the task, dataset, and training configuration.

## What This Project Explores

* LSTM architecture and recurrent computation
* Embedding layers and text representation
* Hidden states and cell states
* Forget, input, and output gates
* Sequence classification
* Backpropagation Through Time
* Padding and packed sequences
* Training loss and evaluation metrics
* Overfitting and generalization

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Datasets

The project is designed to run on a CPU and does not require a GPU.

## 🎯 Learning Objective

This project continues the series by moving from vanilla RNNs to recurrent networks with gated memory.

The focus is on understanding how an LSTM regulates information flow across a sequence and examining whether its additional mechanisms improve performance on a real-world text classification task.

The aim is to understand both sides:

* **How gated memory changes recurrent computation.**
* **Why more sophisticated architectures can still overfit and struggle to generalize.**

## 📚 About the Series

**10 Days of Modeling AI for Beginners**

A hands-on series exploring AI concepts through small, understandable projects.

The objective is to learn by building, experimenting, and examining the mechanisms behind AI—not simply by importing a library and accepting its output.

### What's Next?

**Day 10 — Transformer**

We'll explore attention-based architectures and how they process relationships between tokens without relying on the same recurrent computation used by RNNs and LSTMs.
