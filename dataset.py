import json
from torch.utils.data import Dataset
import torch
class NLPCCTaskDataSet(Dataset):
    def __init__(self, filepath='', is_train=True, mini_test=True, is_test=False):
        self.mini_test = mini_test
        self.is_test = is_test
        self.reply_lens = []
        self.dataset = self.load_json_data(filepath)

    def load_json_data(self, filename):
        return json.load(open(filename,'r',encoding='utf-8'))

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

