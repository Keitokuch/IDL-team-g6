import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from constant import LABEL_LIST
from utils import *


class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels=129, out_channels=129, kernel_size=5):
        nn.Module.__init__(self)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class LinearDOReLU(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        nn.Module.__init__(self)
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.dropout(self.linear(x)))

class SpeakerNet(nn.Module):
    def __init__(self, conv_channels, lstm_hidden, lstm_layers, 
                 conv_size=5, dropout=0.0, pool_size=4, pool_stride=2,
                 input_size=129, num_classes=len(LABEL_LIST)):
        super().__init__()
        conv_shape = [1] + conv_channels
        self.cnn = nn.Sequential(
           *[layer for ic, oc in zip(conv_shape, conv_shape[1:])
                       for layer in [ConvBlock(ic, oc, conv_size), nn.MaxPool2d(pool_size, pool_stride)]])
        self.cnn = nn.Sequential(
            *[ConvBlock(ic, oc, conv_size) for ic, oc in zip(conv_shape, conv_shape[1:])])
        cnn_size = input_size
        for _ in conv_channels:
           cnn_size = (cnn_size - pool_size)//pool_stride + 1
        self.lstm = nn.LSTM(input_size=cnn_size*conv_shape[-1], hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = LinearDOReLU(lstm_layers*2*lstm_hidden, 10*num_classes, dropout)
        self.fc2 = nn.Linear(10*num_classes, num_classes)

    def forward(self, x, x_lens):
        x = self.cnn(x[:, None, :, :])
        x = x.transpose(1, 2).flatten(2, 3)
        # x = rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(x)
        out = hn.transpose(0, 1).flatten(1, 2)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class SpeakerNet1d(nn.Module):
    def __init__(self, conv_channels, lstm_hidden, lstm_layers, 
                 conv_size=5, dropout=0.0, pool_size=4, 
                 pool_stride=2, input_size=129, num_classes=len(LABEL_LIST)):
        super().__init__()
        conv_shape = [input_size] + conv_channels
        self.cnn = nn.Sequential(
            *[layer for ic, oc in zip(conv_shape, conv_shape[1:])
                        for layer in [ConvBlock1d(ic, oc, conv_size), nn.MaxPool1d(pool_size, pool_stride)]])
        self.lstm = nn.LSTM(input_size=conv_shape[-1], hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = LinearDOReLU(lstm_layers*2*lstm_hidden, 10*num_classes, dropout)
        self.fc2 = nn.Linear(10*num_classes, num_classes)

    def forward(self, x, x_lens):
        x = self.cnn(x.transpose(1, 2))
        x = x.transpose(1, 2)
        # x = rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        o, (hn, _) = self.lstm(x)
        out = hn.transpose(0, 1).flatten(1, 2)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
