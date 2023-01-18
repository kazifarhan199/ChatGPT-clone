#!/usr/bin/env python

# Starter code from -> https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
import re
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from collections import Counter
from torch.utils.data import DataLoader

if torch.cuda.is_available(): 
    device = torch.device("cuda:0")


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 100
        self.embedding_dim = 100
        self.num_layers = 1
        self.dataset = dataset

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).data.to(device),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).data.to(device),
       )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        sequence_length,
        num_line,
    ):
        '''As this runs only once it should not be that big of a deal'''
        self.file_path = file_path
        self.num_line = num_line
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.words_indexes = torch.tensor(self.words_indexes).data.to(device)
        
    def load_words(self):
        with open(self.file_path, encoding='utf-8-sig', errors='ignore') as f:
            data = f.read()
        if self.num_line:
            text = self.text_cleaner(data[:self.num_line])
        else:
            text = self.text_cleaner(data)
        return text.split()
    
    def text_cleaner(self, text):
        text = text.lower()
        newString = re.sub(r"'s\b","", text)
        # remove punctuations
        newString = re.sub("[^a-zA-Z]", " ", newString) 
        long_words=[]
        # remove short word
        for i in newString.split():
            if len(i)>=3:                  
                long_words.append(i)
        return (" ".join(long_words)).strip()
    
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.words_indexes) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            self.words_indexes[index:index+self.sequence_length],
            self.words_indexes[index+1:index+self.sequence_length+1],
        )  


"""Django's command-line utility for administrative tasks."""
import os
import sys



def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TextGeneratorBot.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
