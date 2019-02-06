import torch.nn as nn
from .encoder_audio import Encoder


class RNN(nn.Module):
    def __init__(self, feat_length, enc_code_size, rnn_code_size, rate, n_layers=2, init_kernel=None,
                 init_stride=None):
        super(RNN, self).__init__()
        self.audio_feat_samples = int(rate * feat_length)
        self.enc_code_size = enc_code_size
        self.rnn_code_size = rnn_code_size
        self.encoder = Encoder(self.enc_code_size, rate, feat_length, init_kernel=init_kernel,
                               init_stride=init_stride)
        self.rnn = nn.GRU(self.enc_code_size, self.rnn_code_size, n_layers, batch_first=True)

    def forward(self, x, lengths):
        seq_length = x.size()[1]
        x = x.view(-1, 1, self.audio_feat_samples)
        x = self.encoder(x)
        x = x.view(-1, seq_length, self.enc_code_size)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, h = self.rnn(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x.contiguous()
