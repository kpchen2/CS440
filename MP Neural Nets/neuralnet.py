# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np
import numpy.linalg as la

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.lrate = lrate

        h1 = 50
        h2 = 50
        self.neural_net = nn.Sequential(nn.Linear(in_size, h1), nn.ReLU(), nn.Linear(h1,h2), nn.LeakyReLU(), nn.Linear(h2, out_size))

        self.conv = nn.Sequential(nn.Conv2d(3, 50, 3), nn.LeakyReLU())
        self.pool = nn.Sequential(nn.MaxPool2d(6,6), nn.ReLU())
        
        self.optimizer = optim.SGD(self.neural_net.parameters(), self.lrate)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = x.view(-1, 3, 31, 31)
        x_0 = x.shape[0]

        # conv --> pool --> connect
        y = self.conv(x)
        y = self.pool(y)
        y = y.view(x_0,-1)

        return self.neural_net(y)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses = []
    yhats = []

    lrate = 0.07
    loss_func = nn.CrossEntropyLoss()
    in_size = 800
    out_size = 4
    net = NeuralNet(lrate, loss_func, in_size, out_size)

    # data standardization
    s_train_set = (train_set-train_set.mean()) / train_set.std()
    s_dev_set = (dev_set-dev_set.mean()) / dev_set.std()

    # training
    training_dataset = get_dataset_from_arrays(s_train_set, train_labels)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(DataLoader(training_dataset, batch_size, False), 0):
            inputs = data["features"]
            labels = data["labels"]
            # print(inputs)
            # print(labels)

            loss = net.step(inputs, labels)

            # print(loss)
            running_loss += loss

        losses.append(running_loss)

    # development
    for i in s_dev_set:
        y = net.forward(i)
        y = torch.argmax(y)
        yhats.append(y.item())

    # print(type(losses))
    # print(type(yhats))
    # print(type(net))
    return losses, np.array(yhats), net
