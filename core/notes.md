# LSTM — IMDb Sentiment Classification

## 1. Dataset

This project uses the **IMDb Large Movie Review Dataset** to classify movie reviews as positive or negative.

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

1. Convert reviews to lowercase.
2. Tokenize the text into words.
3. Build a vocabulary using training reviews only.
4. Convert words into integer token IDs.
5. Replace unknown words with `<UNK>`.
6. Pad shorter reviews and truncate longer ones to 80 tokens.

**Special tokens:**

* `<PAD>` → `0`
* `<UNK>` → `1`

**Input shape:** `(batch_size, sequence_length)`

For a batch of 32 reviews:

```text
(32, 80)
```

---

## 2. Model Architecture

The model consists of three main components: an **Embedding Layer, an LSTM Layer, and a Linear Output Layer**.

```text
       Movie Review
             |
             v
          Token IDs
             |
             v
       Embedding Layer
             |
             v
          LSTM Layer
          /        \
         h          c
         |          |
         v          v
  Final Hidden   Final Cell
     State         State
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

The LSTM maintains two internal states: a hidden state and a cell state. The final hidden state is used for classification.

### A. Embedding Layer

The embedding layer converts token IDs into learnable, dense vector representations.

```python
nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=64,
    padding_idx=0
)
```

With a vocabulary of 10,000 tokens and an embedding dimension of 64, the embedding matrix has the shape `(10000, 64)`.

For a batch of 32 reviews:

```text
Input:     (32, 80)
Embedded:  (32, 80, 64)
```

The embeddings are learned during training rather than loaded from a pretrained model.

### B. Long Short-Term Memory Layer

The LSTM processes the embedded sequence one timestep at a time. It maintains a hidden state and a cell state, using gates to regulate information flow.

**LSTM configuration:**

| Parameter            |       Value |
| -------------------- | ----------: |
| Input size           |          64 |
| Hidden size          |          64 |
| Number of layers     |           1 |
| Batch format         | Batch first |
| Recurrent activation |        Tanh |
| Gate activation      |     Sigmoid |

#### Forget Gate

Determines how much information from the previous cell state should be retained.

$$
f_t = \sigma(W_f[h_{t-1},x_t]+b_f)
$$

#### Input Gate

Controls how much new information should enter the cell state.

$$
i_t = \sigma(W_i[h_{t-1},x_t]+b_i)
$$

The candidate cell-state update is:

$$
\tilde{c}_t = \tanh(W_c[h_{t-1},x_t]+b_c)
$$

The cell state is updated by combining retained information with selected new information.

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

#### Output Gate

Controls how much of the cell state contributes to the hidden state.

$$
o_t = \sigma(W_o[h_{t-1},x_t]+b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

Where:

* \(x_t\): Input embedding at timestep \(t\).
* \(h_t\): Current hidden state.
* \(h_{t-1}\): Previous hidden state.
* \(c_t\): Current cell state.
* \(c_{t-1}\): Previous cell state.
* \(\sigma\): Sigmoid activation.
* \(\odot\): Element-wise multiplication.

The sigmoid gates produce values between 0 and 1, regulating how information flows through the network.

**Key distinction:** The cell state provides a pathway for retaining and updating information, while the hidden state provides a representation for subsequent computation.

### C. Linear Output Layer

The linear layer maps the final hidden representation to a single sentiment logit.

```python
nn.Linear(64, 1)
```

For a batch of 32 reviews:

```text
Final hidden state: (32, 64)
Output logits:      (32,)
```

The output logit can be converted into a probability using the sigmoid function:

$$
p = \frac{1}{1+e^{-z}}
$$

Where \(z\) is the output logit and \(p\) represents the predicted probability of a positive review.

**Classification rule:**

* Probability ≥ 0.5: Positive
* Probability < 0.5: Negative

---

## 3. How Does the Model Learn?

The model uses **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)** for binary sentiment classification.

This loss function combines sigmoid activation and binary cross-entropy into a numerically stable calculation.

For a single example, the binary cross-entropy expression is:

$$
L = -[y\log(p)+(1-y)\log(1-p)]
$$

Where:

* \(y\): Actual label (0 or 1).
* \(p\): Predicted probability of the positive class.

### Training Process

1. The model receives a batch of tokenized reviews.
2. The embedding layer converts token IDs into vectors.
3. The LSTM processes the sequence and updates its hidden and cell states.
4. The final hidden state is passed to the linear layer.
5. The loss function measures the prediction error.
6. Backpropagation calculates gradients through the output layer, LSTM, and embedding layer.
7. The optimizer updates the model parameters.

Because the recurrent computation is unfolded across timesteps during gradient calculation, this training process is known as **Backpropagation Through Time (BPTT)**.

### Padding Handling

Reviews have different lengths, so shorter reviews are padded.

Our implementation uses `pack_padded_sequence` to prevent the LSTM from processing trailing padding positions as actual review tokens.

This allows the final hidden state to correspond to the end of the actual review rather than additional recurrent steps over padding.

---

## 4. Experimental Results

The model was trained for 10 epochs using the IMDb dataset.

| Metric                  |     Result |
| ----------------------- | ---------: |
| Maximum sequence length |  80 tokens |
| Embedding dimension     |         64 |
| Hidden size             |         64 |
| Batch size              |         32 |
| Learning rate           |      0.001 |
| Optimizer               |       Adam |
| Epochs                  |         10 |
| Best test accuracy      | **78.88%** |
| Best epoch              |          5 |
| Final training accuracy |     96.04% |
| Final test accuracy     |     78.04% |

Training accuracy continued to increase, while test loss generally worsened after the earlier epochs.

This suggests **overfitting**: the model continued fitting the training data without achieving corresponding improvements on unseen reviews.

The best test accuracy occurred at Epoch 5 rather than Epoch 10.

These results describe one experiment and do not guarantee the same performance across different runs or configurations.

---

## 5. Limitations

### Sequential Computation

Each timestep depends on the preceding recurrent state. Consequently, recurrent computations within a sequence cannot generally be parallelized across timesteps in the same way as independent feed-forward operations.

### Computational Complexity

LSTMs use additional gates and maintain both hidden and cell states, making them more computationally expensive than basic vanilla RNNs.

### Long-Term Dependencies

LSTMs are designed to help preserve information across many timesteps, but they do not guarantee that every important part of a long sequence will be retained.

### Padding and Truncation

Reviews are truncated to a maximum length of 80 tokens. Information beyond that limit is discarded, while shorter reviews are padded. Packed-sequence handling prevents trailing padding from being processed as real tokens.

### Overfitting

The training accuracy reached 96.04%, while the final test accuracy was 78.04%. This gap suggests that the model learned patterns that did not generalize equally well to unseen reviews.

### Dependence on Training Data

The model learns patterns from the training reviews. Its performance on unseen data depends on how well those patterns generalize.

---

## 6. Learning Objectives

This project demonstrates how an LSTM processes sequential data using gated recurrent computation.

Key concepts include:

* **Embeddings:** Learning vector representations of tokens.
* **Cell state:** Maintaining an internal pathway for information across timesteps.
* **Forget gate:** Regulating the retention of existing information.
* **Input gate:** Regulating the incorporation of new information.
* **Output gate:** Controlling the hidden representation.
* **Sequence classification:** Mapping an entire review to a sentiment prediction.
* **Backpropagation Through Time:** Training recurrent computations through gradient propagation.

> The central idea is that **LSTM introduces gated memory to help regulate information flow across a sequence**, addressing some of the learning difficulties associated with vanilla RNNs.
