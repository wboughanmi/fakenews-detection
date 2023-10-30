import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True)

        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, seq):
        embed = self.embedding(seq)

        output, _ = self.lstm(embed)

        output = self.linear(output[-1])
        output = self.sigmoid(output)

        return output


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=256) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=0)

        self.lstm = nn.GRU(input_size=embedding_size, hidden_size=hidden_size)

        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        embed = self.embedding(seq)

        output, _ = self.lstm(embed)

        output = self.linear(output[-1])
        output = self.sigmoid(output)

        return output