from transformers import AdamW, BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import copy
import warnings
import os
import pickle


def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class GRU(nn.Module):
    def __init__(self, **config):
        super().__init__()
        vocab_size = config.get('vocab_size')
        dropout = config.get('dropout', 0.4)
        d_model = config.get('d_model', 256)
        num_layers = config.get('num_layers', 1)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, bidirectional=True, batch_first=True)

        init_params(self.embedding)
        init_params(self.gru)

        self.d_model = d_model * 2

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(0).detach()
        E = self.embedding_dropout(self.embedding(input_ids)).transpose(0, 1)
        H, h1 = self.gru(E)
        H = H.transpose(0, 1)
        h = universal_sentence_embedding(H, attention_mask)
        return h


class GRUAttention(nn.Module):
    def __init__(self, **config):
        super().__init__()
        vocab_size = config.get('vocab_size')
        dropout = config.get('dropout', 0.4)
        d_model = config.get('d_model', 256)
        num_layers = config.get('num_layers', 1)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.w = nn.Linear(2 * d_model, 1)

        init_params(self.embedding)
        init_params(self.gru)

        self.d_model = d_model * 2

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(0).detach()
        E = self.embedding_dropout(self.embedding(input_ids)).transpose(0, 1)
        H, h1 = self.gru(E)
        H = H.transpose(0, 1)  # bc_size, len, d_model
        wh = self.w(H).squeeze(2)  # bc_size, len
        # print(wh.size())
        attention = F.softmax(F.tanh(wh).masked_fill(mask=~attention_mask, value=-np.inf)).unsqueeze(1)
        # bc_size, 1, len

        presentation = torch.bmm(attention, H).squeeze(1)  # bc_size, d_model
        return presentation


class Hierarchical(nn.Module):
    def __init__(self, backbone, class_num):
        super().__init__()
        self.drop_out = nn.Dropout(0.4)
        self.private = nn.ModuleList([copy.deepcopy(backbone) for num in class_num])
        d_model = backbone.d_model

        self.class_num = class_num
        self.gru = nn.ModuleList(
            [nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True) for num in class_num])
        self.linear = nn.ModuleList([nn.Linear(d_model, num) for num in class_num])
        for layer in self.linear:
            init_params(layer)
        for layer in self.gru:
            init_params(layer)

    def forward(self, input_ids, **kwargs):
        bc_size, dialog_his, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = input_ids.ne(0).detach()

        res = []
        for private_module, gru, cls_layer in zip(self.private, self.gru, self.linear):
            private_out = private_module(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            private_out = private_out.view(bc_size, dialog_his, -1)  # bc_size, dialog_his, d_model
            H, hidden = gru(private_out)
            hidden = hidden.squeeze(0)  # bc_size, d_model
            hidden = self.drop_out(hidden)
            rep = hidden
            res.append(cls_layer(rep))
        return res


class HierarchicalAttention(nn.Module):
    def __init__(self, backbone, class_num):
        super().__init__()
        self.drop_out = nn.Dropout(0.4)
        self.private = nn.ModuleList([copy.deepcopy(backbone) for num in class_num])
        d_model = backbone.d_model

        self.w = nn.ModuleList([nn.Linear(d_model, 1) for num in class_num])

        self.class_num = class_num
        self.gru = nn.ModuleList(
            [nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True) for num in class_num])
        self.linear = nn.ModuleList([nn.Linear(d_model, num) for num in class_num])
        for layer in self.linear:
            init_params(layer)
        for layer in self.gru:
            init_params(layer)
        for layer in self.w:
            init_params(layer)

    def forward(self, input_ids, **kwargs):
        bc_size, dialog_his, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = input_ids.ne(0).detach()

        res = []
        for private_module, gru, w, cls_layer in zip(self.private, self.gru, self.w, self.linear):
            private_out = private_module(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            private_out = private_out.view(bc_size, dialog_his, -1)  # bc_size, dialog_his, d_model
            H, hidden = gru(private_out)
            # H = H.transpose(0, 1)  # bc_size, dialog_his, d_model
            wh = w(H).squeeze(2)  # bc_size, dialog_his
            attention = F.softmax(F.tanh(wh)).unsqueeze(1)  # bc_size, 1, dialog_his
            hidden = torch.bmm(attention, H).squeeze(1)  # bc_size, d_model

            hidden = self.drop_out(hidden)
            rep = hidden
            res.append(cls_layer(rep))
        return res


class BERTBackbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        name = config.get('name', 'bert-base-chinese')
        self.layers_used = config.get('layers_used', 2)
        self.bert = BertModel.from_pretrained(name, output_hidden_states=True, output_attentions=True)
        self.d_model = 768 * self.layers_used * 2

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_out = out[2]
        out = [bert_out[-i - 1] for i in range(self.layers_used)]
        out = torch.cat(out, dim=-1)
        out = universal_sentence_embedding(out, attention_mask)

        cls = [bert_out[-i - 1].transpose(0, 1)[0] for i in range(self.layers_used)]
        cls = torch.cat(cls, dim=-1)
        out = torch.cat([cls, out], dim=-1)
        return out


class ClassModel(nn.Module):
    def __init__(self, backbone, class_num):
        super().__init__()
        self.drop_out = nn.Dropout(0.4)
        self.private = nn.ModuleList([copy.deepcopy(backbone) for num in class_num])
        d_model = backbone.d_model
        self.class_num = class_num
        self.linear = nn.ModuleList([nn.Linear(d_model, num) for num in class_num])
        for layer in self.linear:
            torch.nn.init.normal_(layer.weight, std=0.02)

    def forward(self, input_ids, **kwargs):
        input_ids = input_ids
        attention_mask = input_ids.ne(0).detach()
        res = []
        for private_module, cls_layer in zip(self.private, self.linear):
            private_out = private_module(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            rep = private_out
            rep = self.drop_out(rep)
            res.append(cls_layer(rep))
        return res
