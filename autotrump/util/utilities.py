import json
import math
import string
import time

import torch
import unidecode
from torch.autograd import Variable

all_characters = string.printable
n_characters = len(all_characters)


def combine_tweets():
    super_data = []
    dates = ['09', '10', '11', '12', '13', '14', '15', '16', '17']
    for date in dates:
        with open('../res/condensed_20' + date + '.json') as data_file:
            data = json.load(data_file)
            super_data.extend(data)
    with open('../res/trump_tweets.json', 'w') as data_file:
        json.dump(super_data, data_file)


def load_tweets(fp='../res/trump_tweets.json'):
    return json.load(read_file(fp))


def save_tweets_text():
    tweets = load_tweets()
    with open('../res/trump_tweets.txt', 'w') as data_file:
        for tweet in tweets:
            data_file.write('%s\n\n' % unidecode.unidecode(tweet['text']).replace('\n', ''))


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def char_tensor(s):
    tensor = torch.zeros(len(s)).long()
    for c in range(len(s)):
        tensor[c] = all_characters.index(s[c])
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
