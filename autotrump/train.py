import os
import random

from torch import nn

from generator import *
from model import RNN


class Train(object):
    def __init__(self, training_file='../res/trump_tweets.txt', model_file='../res/model.pt', n_epochs=1000000,
                 hidden_size=256, n_layers=2, learning_rate=0.001, chunk_len=140):
        self.training_file = training_file
        self.model_file = model_file
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.chunk_len = chunk_len
        self.file, self.file_len = read_file(training_file)
        if os.path.isfile(model_file):
            self.decoder = torch.load(model_file)
            print('Loaded old model!')
        else:
            self.decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
            print('Constructed new model!')
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.generator = Generator(self.decoder)

    def train(self, inp, target):
        hidden = self.decoder.init_hidden()
        self.decoder.zero_grad()
        loss = 0
        for c in range(self.chunk_len):
            output, hidden = self.decoder.forward(inp[c], hidden)
            loss += self.criterion(output, target[c])
        loss.backward()
        self.decoder_optimizer.step()
        return loss.data[0] / self.chunk_len

    def save(self):
        torch.save(self.decoder, self.model_file)
        print('Saved as %s' % self.model_file)

    def random_training_set(self, chunk_len):
        start_index = random.randint(0, self.file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = self.file[start_index:end_index]
        inp = char_tensor(chunk[:-1])
        target = char_tensor(chunk[1:])
        return inp, target

    def start(self):
        start_time = time.time()
        print("Training for %d epochs..." % self.n_epochs)
        best_loss = None
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(*self.random_training_set(self.chunk_len))
            if not best_loss or loss < best_loss:
                self.save()
                best_loss = loss
                print('[%s (%d %d%%) %.4f]' % (time_since(start_time), epoch, epoch / self.n_epochs * 100, loss))
                print(self.generator.generate(), '\n')
        print("Finished training, saving...")
        self.save()
