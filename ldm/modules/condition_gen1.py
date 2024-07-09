import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import argparse


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.emb = nn.Embedding(max_seq_length, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class YearPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length=24):
        super(YearPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create a positional encoding matrix
        self.position = torch.arange(0, max_seq_length).unsqueeze(1).cuda()
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)).cuda()
        self.pe = torch.zeros(max_seq_length, d_model).cuda()
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(-1)
        pe_s = self.pe[x, :]
        pe_s = pe_s.view(batch_size, seq_len, -1)
        return pe_s.cuda()


class SequenceEncoder(nn.Module):
    def __init__(self, seq_len=6, d_model=768, nhead=8, num_layers=3):
        super(SequenceEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.pos_emb = YearPositionalEmbedding(d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)).cuda()

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear2 = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x, real_times):

        time_embs = self.pos_emb(real_times)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) + x[:, 5:6, :]

        x = torch.cat((x[:, :5, :], cls_tokens), dim=1)

        x = self.dropout1(x + time_embs)

        x = self.trans_encoder(x)

        x = self.linear1(x[:, -1, :])
        x = self.linear2(self.dropout2(x))

        return x


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super(ImageEncoder, self).__init__()
        self.base_network = models.resnet18(pretrained=True)
        self.base_network.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_network.fc = nn.Linear(in_features=512, out_features=out_dim, bias=True)
        nn.init.kaiming_normal_(self.base_network.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal_(self.base_network.fc.weight, a=0, mode='fan_out')
        nn.init.normal_(self.base_network.fc.bias, std=1e-6)

    def forward(self, x):
        x = self.base_network(x)
        return x


class ConditionGen(nn.Module):
    def __init__(self):
        super(ConditionGen, self).__init__()

        self.seq_len = 6
        self.d_model = 768
        self.nhead = 8
        self.num_layers = 3

        self.image_encoder = ImageEncoder(
            out_dim=self.d_model,
        )
        self.sequence_encoder = SequenceEncoder(
            seq_len=self.seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
        )

    def forward(self, x, real_times):
        batch_size = x.shape[0]

        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.image_encoder(x)
        x = x.view(batch_size, self.seq_len, -1)

        x = self.sequence_encoder(x, real_times)
        
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224)
    cfg = parser.parse_args()
