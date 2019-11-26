from torch import nn


class IMDBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim,  hidden_dim, num_layers, lstm_dropout, fc_dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)

        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, h):
        batch_size = x.size(0)

        x = self.embed(x)
        x, h = self.lstm(x, h)
        x = x.contiguous().view(-1, self.hidden_dim)

        return self.fc(self.dropout(x)).view(batch_size, -1)[:, -2:], h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to('cuda'),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to('cuda'))

        return hidden

