# Training Notes

## 1. Training Configuration

The model is trained on the IMDb dataset for binary sentiment classification.

| Parameter               |             Value |
| ----------------------- | ----------------: |
| Dataset                 |              IMDb |
| Training samples        |            25,000 |
| Testing samples         |            25,000 |
| Maximum vocabulary size |            10,000 |
| Maximum sequence length |         80 tokens |
| Embedding dimension     |                64 |
| Hidden size             |                64 |
| Number of LSTM layers   |                 1 |
| Batch size              |                32 |
| Epochs                  |                10 |
| Learning rate           |             0.001 |
| Optimizer               |              Adam |
| Loss function           | BCEWithLogitsLoss |
| Device                  |               CPU |

## 2. Training Pipeline

The training process follows these steps:

1. Load the IMDb dataset and build a vocabulary from the training reviews.
2. Convert reviews into token IDs and prepare batches.
3. Pass token IDs through the embedding layer.
4. Process the embedded sequences using the LSTM.
5. Use the final hidden state to generate sentiment logits.
6. Calculate the loss against the actual labels.
7. Perform backpropagation and update parameters using Adam.
8. Evaluate the model on the test dataset after each epoch.

### Loss Function

We use `BCEWithLogitsLoss`, which combines sigmoid activation and binary cross-entropy in a numerically stable implementation.

The model returns raw logits during training.

### Evaluation Metrics

* **Training loss:** Measures prediction error on training data.
* **Training accuracy:** Measures the proportion of correct training predictions.
* **Test loss:** Measures prediction error on unseen test data.
* **Test accuracy:** Measures the proportion of correct test predictions.

The test set is used for evaluation after each epoch. Selecting a checkpoint based on test performance means the test set influences model selection; a separate validation set would provide a cleaner final evaluation.

## 3. Observations

### Learning Progress

Training loss decreased consistently, while training accuracy increased from 58.02% to 96.04%.

Test accuracy improved rapidly during the first five epochs, reaching its highest value of **78.88% at Epoch 5**.

### Overfitting

After the earlier epochs, training performance continued improving while test loss generally increased.

This indicates that the model was fitting the training data more closely without achieving corresponding improvements on unseen reviews.

### Accuracy vs. Loss

Test accuracy remained relatively stable after Epoch 5, but test loss increased.

This can happen when predictions become more confident without becoming more accurate overall. Incorrect predictions with high confidence receive a larger penalty from cross-entropy loss.

### Best Checkpoint

The highest observed test accuracy occurred at Epoch 5, while the lowest test loss occurred at Epoch 4.

These metrics measure different aspects of performance, so their best epochs do not necessarily coincide.

## 4. Key Takeaways

* Training accuracy alone is insufficient to assess model quality.
* More epochs do not necessarily improve generalization.
* Test loss can worsen even when test accuracy changes very little.
* The best checkpoint may occur before the final epoch.
* LSTM performance depends on the dataset, preprocessing, architecture, and training configuration.

**Result:** The model achieved a best observed test accuracy of **78.88%** during this 10-epoch experiment.
