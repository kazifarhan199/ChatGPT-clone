import re
import torch
import numpy as np
from torch import nn, optim
from collections import Counter
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """The LSTM models we will use"""
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        self.lstm_size = 100
        self.embedding_dim = 100
        self.num_layers = 1
        self.n_vocab = n_vocab

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

def text_cleaner(text):
    # lower case text
    newString = re.sub(r"'s\b","", text)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()


def train(dataset, model, batch_size, max_epochs, sequence_length):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

        print({ 'epoch': epoch, 'loss': loss.item(), 'perplexity': torch.exp(loss).item()})
            
        if loss.item() < 0.2:
            break


def predict(dataset, model, text, next_words=100):
    text = text_cleaner(text)
    words = text.split()
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().to('cpu').numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


def generate_text(text, length):
    try:
        return ' '.join(predict(dataset, loded_model, text, next_words=length))
    except:
        return "! Can't generate text for the given sequence, some words are not present in the word dictnary"


file_path = "data/smalltext.train.txt"
sequence_length = 4
dataset = Dataset(file_path, sequence_length, num_line=200_000)
n_vocab = len(dataset.uniq_words)
model_path = "model_dict.pkl"

if __name__ == "__main__":

    print("Initializing model...")
    max_epochs = 20
    batch_size = 512
    model = Model(n_vocab).to(device)
    
    print("Training model...")
    train(dataset, model, batch_size, max_epochs, sequence_length)

    print("Saving model ")
    torch.save(model.state_dict(), model_path)
else:
    print("Loading model...")
    loded_model = Model(n_vocab).to(device)
    loded_model.load_state_dict(torch.load(model_path))
    loded_model.eval()