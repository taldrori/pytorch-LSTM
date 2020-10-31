from torchfile import load
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.linear(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def train(data, seq_size, model, num_epochs, optimizer, criterion):
    running_loss = []
    for epoch in range(num_epochs):

        # random starting point (1st 100 chars) from data to begin
        data_ptr = np.random.randint(100)
        hidden_state = None
        epoch_loss = 0.0
        n = 0

        while True:
            input_seq = data[data_ptr: data_ptr + seq_size]
            target_seq = data[data_ptr + 1: data_ptr + seq_size + 1]
            # TODO implement this in a less stupid way
            x = np.zeros((seq_size, 35))
            y = np.zeros((seq_size, 35))
            for i, d in enumerate(input_seq):
                x[(i, d - 1)] = 1
            for i, d in enumerate(target_seq):
                y[(i, d - 1)] = 1
            x = torch.tensor(x).long()
            y = torch.tensor(y).long()
            input_seq = x
            target_seq = y

            # forward pass
            output, hidden_state = model(input_seq, hidden_state)

            # compute loss
            loss = criterion(output, target_seq)

            # loss = criterion(torch.squeeze(output), torch.squeeze(target_seq))
            epoch_loss += loss.item()

            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the data pointer
            data_ptr += seq_size
            n += 1

            # if at end of data : break
            if data_ptr + seq_size + 1 > len(data):
                break

        running_loss.append(epoch_loss / n)
        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch + 1, running_loss[-1]))

        # sample / generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = None

        # random character from data to begin
        rand_index = np.random.randint(len(data) - 1)
        input_seq = data[rand_index: rand_index + 1]

        print("----------------------------------------")
        while True:
            # forward pass
            output, hidden_state = model(input_seq, hidden_state)

            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            # print the sampled character
            print(mapping[index.item() + 1].decode('utf-8'), end='')

            # next input is current output
            input_seq[0][0] = index.item()
            data_ptr += 1

            if data_ptr > op_seq_len:
                break

        print("\n----------------------------------------")


def load_data():
    train = load('practical6-data/train.t7')
    vocab = load('practical6-data/vocab.t7')
    mapping = {y: x for x, y in vocab.items()}
    return train, mapping


data, mapping = load_data()
data = list(data)

# data tensor on device
data = torch.tensor(data).long()
data = torch.unsqueeze(data, dim=1)

hidden_size = 10  # 512
seq_size = 500
num_layers = 2  # 3
lr = 0.002
epochs = 2
op_seq_len = 200

model = Net(35, 35, hidden_size, num_layers)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 3

train(data, seq_size, model, num_epochs, optimizer, criterion)
