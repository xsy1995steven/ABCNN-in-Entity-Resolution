import torch
import torch.nn as nn
import torch.nn.functional as F


class attr_sum_simple_cnn(nn.Module):
    def __init__(self):
        super(attr_sum_simple_cnn, self).__init__()
        self.sentence_length = 36
        self.CNN1 = Net()
        self.CNN2 = Net()

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x1 = F.pad(x1, (0, 0, 0, self.sentence_length - x1.size()[2]))
        x2 = F.pad(x2, (0, 0, 0, self.sentence_length - x2.size()[2]))
        x1 = self.CNN1(x1)
        x2 = self.CNN2(x2)
        return x1, x2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        words_dim = 300
        self.conv1 = nn.Conv2d(
                in_channels=1,  # input height
                out_channels=6,  # n_filters
                kernel_size=(3, words_dim),  # filter size
                stride=1,  # filter movement/step
                padding=(2,0),#padding=(kernel_size-1)/2
            )
        self.conv2 = nn.Conv2d(
            in_channels=1,  # input height
            out_channels=6,  # n_filters
            kernel_size=(4, words_dim),  # filter size
            stride=1,  # filter movement/step
            padding=(3, 0),  # padding=(kernel_size-1)/2
        )
        self.conv3 = nn.Conv2d(
            in_channels=1,  # input height
            out_channels=6,  # n_filters
            kernel_size=(5, words_dim),  # filter size
            stride=1,  # filter movement/step
            padding=(4, 0),  # padding=(kernel_size-1)/2
        )# output = [batch, output_channel, sql_len, 1]

    def forward(self, x):
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * Ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # (batch, channel_output) * Ks
        x = torch.cat(x, 1)
        return x
