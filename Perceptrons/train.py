#!/usr/bin/env python


'''This program uses perceptron learning algorithm to train perceptrons
that will learn to classify the handwritten digits in the MNIST dataset.'''

import csv
import random
import numpy
import matplotlib.pyplot as plt

train_file = 'mnist_train.csv'
test_file = 'mnist_test.csv'

SIZE_TRAIN = 60000
SIZE_TEST = 10000
SIZE_INPUT = 785
EPOCH = 50

lr = [0.001, 0.01, 0.1]

train_set = []
test_set = []

def read_data(input_file):
  data = numpy.genfromtxt(input_file, delimiter = ',')

  return data

def preProcessing(data, size):
  for i in range(0, size):
    for j in range(1, SIZE_INPUT):
      data[i][j] = data[i][j] / 255.0

def init_weights(size):
  weights = numpy.zeros((10, size))

  for i in range(10):
    for j in range(size):
      weights[i][j] = random.uniform(-0.05, 0.05)

  return weights

def perceptronTrain(eta, weights):
    for i in range(SIZE_TRAIN):
      t_raw = train_set[i][0]
      t = numpy.zeros(10)
      for j in range(10):
        if t_raw == j:
          t[j] = 1
        else:
          t[j] = 0
        # compute y = (sum(wi * xi) > 0)
        y_raw = weights[j][0] + numpy.dot(weights[j], train_set[i]) \
                - weights[j][0] * train_set[i][0]
        if y_raw > 0:
           y = 1
        else:
           y = 0
        #compute delta_weight = eta * (t - y) * xi
        delta_weight = eta * (t[j] - y) * numpy.array(train_set[i])
        delta_weight[0] = eta * (t[j] - y) * 1
        # update weights
        weights[j] += delta_weight

# compute accuracy
def computeAccuracy(dataSet, size, weights):
  pv = numpy.zeros(10)
  correct = 0

  for i in range(size):
    for digit in range(10):
      pv[digit] = numpy.dot(weights[digit], dataSet[i]) - weights[digit][0] \
                  * dataSet[i][0] + weights[digit][0]

    pvMax = pv[0]
    pvMaxIx = 0
    for j in range(1, 10):
      if pvMax < pv[j]:
        pvMax = pv[j]
        pvMaxIx = j

    if dataSet[i][0] == pvMaxIx:
      correct += 1

  accuracy = float(correct) / float(size) * 100
  return accuracy

def visualizeAccuracy(eta, train_accuracy, test_accuracy):
  fig = plt.figure()

  plt.title('Accuracy after each epoch, $\eta$ = %.3f' % eta)
  plt.plot(range(EPOCH+1), train_accuracy, label='Accuracy on the training data')
  plt.plot(range(EPOCH+1), test_accuracy, label='Accuracy on the test data')
  plt.legend(loc=4)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  fig.savefig('plot_eta_%d' % int(eta * 1000))

def mainTrain():
  for eta in lr:
    weights = init_weights(SIZE_INPUT)

    train_accuracy = numpy.zeros(EPOCH + 1)
    test_accuracy = numpy.zeros(EPOCH + 1)

    # traing process, EPOCH iteration in total
    for i in range(EPOCH):
      train_accuracy[i] = computeAccuracy(train_set, SIZE_TRAIN, weights)
      test_accuracy[i] = computeAccuracy(test_set, SIZE_TEST, weights)
      #print('%d %.6f %.6f' % (i, train_accuracy[i], test_accuracy[i]))

      perceptronTrain(eta, weights)

    train_accuracy[EPOCH] = computeAccuracy(train_set, SIZE_TRAIN, weights)
    test_accuracy[EPOCH] = computeAccuracy(test_set, SIZE_TEST, weights)
    #print('%d %.6f %.6f' % (EPOCH, train_accuracy[EPOCH], test_accuracy[EPOCH]))

    visualizeAccuracy(eta, train_accuracy, test_accuracy)

if __name__ == "__main__":
  train_set = read_data(train_file)
  test_set = read_data(test_file)

  preProcessing(train_set, SIZE_TRAIN)
  preProcessing(test_set, SIZE_TEST)

  mainTrain()

