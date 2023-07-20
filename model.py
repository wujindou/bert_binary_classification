#coding:utf-8
import sys
import json
import sys
from transformers import BertModel
from transformers import AutoModel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

class BertClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=config.pretrain_model_path)
        self.dropout = nn.Dropout(0.2)
        self.cls_layer1 = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :] if self.config.use_cls else torch.mean(bert_output.last_hidden_state,dim=1)


        logits = self.dropout(pooled_output)
        output = self.cls_layer1(logits)
        return torch.sigmoid(output)
