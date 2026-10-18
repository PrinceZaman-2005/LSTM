import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_size=64,
        num_layers=1,
        output_size=1,
        dropout=0.0,
    ):
        super().__init__()

        # Convert token IDs into dense vectors
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # LSTM processes the sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Convert the final hidden representation into a prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)

        # Calculate actual review lengths, excluding padding
        lengths = x.ne(0).sum(dim=1).clamp(min=1).cpu()

        # Convert token IDs into embeddings
        embedded = self.embedding(x)

        # Prevent the LSTM from processing trailing padding
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        # Process the sequence
        _, (hidden, cell) = self.lstm(packed)

        # Use the final hidden state of the last LSTM layer
        final_hidden = hidden[-1]

        # Produce a sentiment logit
        logits = self.fc(final_hidden)

        return logits.squeeze(-1)
