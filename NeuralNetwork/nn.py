#!/usr/bin/env python

'''This program implements a two-layer neural network (i.e., one hidden layer)
to perform the handwritten digit recognition in the MNIST dataset. The
experiments includes three parts: 1. vary the number of hidden units; 2. vary
the momentum value; 3. vary the number of training examples.'''

import csv
import random
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
  def __init__(self, tr_file, te_file, tr_size, te_size, in_size, epoch, lr, mem):
    self.train_set = []
    self.test_set = []
    self.train_file = tr_file
    self.test_file = te_file
    self.train_size = tr_size
    self.test_size = te_size
    self.input_size = in_size
    self.epoch = epoch
    self.lr = lr
    self.mem = mem
    self.train_t = np.zeros(tr_size)
    self.test_t = np.zeros(te_size)

  def read_data(self):
    self.train_set = np.genfromtxt(self.train_file, delimiter = ',')
    self.test_set = np.genfromtxt(self.test_file, delimiter = ',')

  def preProcessing(self):
    for i in range(0, self.train_size):
      self.train_t[i] = self.train_set[i][0]
      for j in range(0, self.input_size):
        self.train_set[i][j] = self.train_set[i][j] / 255.0
        self.train_set[i][0] = 1.0

    for i in range(0, self.test_size):
      self.test_t[i] = self.test_set[i][0]
      for j in range(0, self.input_size):
        self.test_set[i][j] = self.test_set[i][j] / 255.0
        self.test_set[i][0] = 1.0

  def init_weights(self, row, column):
    weights = np.zeros((row, column))

    for i in range(row):
      for j in range(column):
        weights[i][j] = random.uniform(-0.05, 0.05)

    return weights

  def mainTrain(self):
    # various number of hidden units
    #hu = [21, 51, 101]
    hu = [21]
    for n in hu:
      self.NNTraining(n)

  def NNTraining(self, hu):
    w_ji = self.init_weights(hu, self.input_size)
    w_kj = self.init_weights(10, hu)

    h = np.zeros(hu)
    output = np.zeros(10)

    dw_kj = np.zeros((10, hu))
    dw_ji = np.zeros((hu, self.input_size))

    train_accuracy = np.zeros(self.epoch + 1)
    test_accuracy = np.zeros(self.epoch + 1)

    # traing process, self.epoch iteration in total
    for n in range(self.epoch):
      train_accuracy[n] = self.computeAccuracy(self.train_set, self.train_t, \
                          self.train_size, hu, w_ji, w_kj)
      test_accuracy[n] = self.computeAccuracy(self.test_set, self.test_t, \
                         self.test_size, hu, w_ji, w_kj)
      print('%d %.6f %.6f' % (n, train_accuracy[n], test_accuracy[n]))

      for k in range(self.train_size):
        # propagate the input forward
        # from the input to the hidden layer
        h[0] = 1.0
        for j in range(1, hu):
          h[j] = sigmoid(np.dot(w_ji[j], self.train_set[k]))

        #from the hidden layer to the output
        for j in range(10):
          output[j] = sigmoid(np.dot(w_kj[j], h))

        # calculate output error terms: delta_k=ok*(1-ok)*(tk-ok)
        error_o = np.zeros(10)
        target = np.zeros(10)
        for j in range(10):
          if self.train_t[k] == j:
            target[j] = 0.9
          else:
            target[j] = 0.1
          error_o[j] = output[j] * (1.0 - output[j]) * (target[j] - output[j])

        # calculate hidden error terms: h*(1-h)*sum(wh*delta_k)
        error_h = np.zeros(hu)
        for j in range(hu):
          error_h[j] = h[j] * (1 - h[j]) * np.dot(w_kj[:,j], error_o)
        # update weights
        # from the hidden layer to the output
        for j in range(10):
          dw_kj[j] *= self.mem
          dw_kj[j] = np.add(dw_kj[j], self.lr * error_o[j] * h)
          w_kj[j] = np.add(w_kj[j], dw_kj[j])
        # from the input to the hidden layer
        for j in range(hu):
          dw_ji[j] *= self.mem
          dw_ji[j] = np.add(dw_ji[j], self.lr * error_h[j] * self.train_set[k])
          w_ji[j] = np.add(w_ji[j], dw_ji[j])

    train_accuracy[self.epoch] = self.computeAccuracy(self.train_set, \
                                 self.train_t, self.train_size, hu, w_ji, w_kj)
    test_accuracy[self.epoch] = self.computeAccuracy(self.test_set, \
                                self.test_t, self.test_size, hu, w_ji, w_kj)
    print('%d %.6f %.6f' % (self.epoch, train_accuracy[self.epoch], \
          test_accuracy[self.epoch]))

    self.visualizeAccuracy(self.lr, train_accuracy, test_accuracy)

  # compute accuracy
  def computeAccuracy(self, data, target, size, hu, w_ji, w_kj):
    correct = 0

    for i in range(size):
      correct += self.classify(data[i], target[i], hu, w_ji, w_kj)

    accuracy = float(correct) / float(size) * 100
    return accuracy

  def classify(self, data, target, hu, w_ji, w_kj):
    h = np.zeros(hu)
    output = np.zeros(10)

    correct = 0

    # propagate the input forward
    # from the input to the hidden layer
    h[0] = 1.0
    for j in range(1, hu):
      h[j] = sigmoid(np.dot(w_ji[j], data))

    #from the hidden layer to the output
    for j in range(10):
      output[j] = sigmoid(np.dot(w_kj[j], h))

    OutMax = output.argmax(axis = 0)
    if target == OutMax:
      correct += 1

    return correct

  def visualizeAccuracy(self, lr, train_accuracy, test_accuracy):
    fig = plt.figure()

    plt.title('Accuracy after each epoch, $\eta$ = %.3f' % lr)
    plt.plot(range(self.epoch+1), train_accuracy, label='Accuracy on the training data')
    plt.plot(range(self.epoch+1), test_accuracy, label='Accuracy on the test data')
    plt.legend(loc=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    fig.savefig('plot_lr_%d' % int(lr * 1000))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

if __name__ == "__main__":
  nn = NeuralNetwork('mnist_train.csv','mnist_test.csv', 60000, 10000, 785, 50, 0.1, 0.9)
  nn.read_data()
  nn.preProcessing()

  nn.mainTrain()
