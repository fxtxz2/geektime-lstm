# 定义模型
import torch


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                                  dropout=dropout_rate, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True,
                                                                  enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction