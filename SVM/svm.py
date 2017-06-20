#!/usr/bin/env python

'''This program does experiments with linear SVMs and feature
   selection.'''

import random
import subprocess
import numpy as np
import operator
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import re


class SVM :
  def __init__(self, data_f):
    self.data_f = data_f

  def training(self):
    # Preprocessing data
    train_data, test_data = self.preprocessData()
    self.save2File(train_data, test_data)

    # 1. Training SVM model
    subprocess.call('./svm_learn train.data svm_model', shell = True)
    subprocess.call('./svm_classify test.data svm_model output > svm_exp1', shell = True)

    self.plotROC()

    # 2. Feature selection with linear SVM
    # Get weight vector using the perl script
    subprocess.call('perl svm2weight.pl svm_model >> weights', shell = True)

    for i in range(2, 58):
      n_train, n_test = self.featureSelector(i, train_data, test_data)
      self.save2File(n_train, n_test)
      subprocess.call('./svm_learn train.data f_svm_model', shell = True)
      subprocess.call('./svm_classify test.data f_svm_model output >> svm_exp2', shell = True)

    accu = self.readAccuracy('svm_exp2')
    self.plotAccuracy(range(2, 58), accu, 'featureAccu.pdf')


  def readAccuracy(self, resFile):
    accu = []
    with open(resFile, 'r') as f:
      lines = f.readlines()

    for line in lines:
      match = re.match('Accuracy on test set: (\d+).(\d+)% \((\d+) correct, (\d+) incorrect, (\d+) total\)', line)
      if match:
        accu.append(float(match.group(1)))

    return accu


  def plotAccuracy(self, n, accu, name):
    plt.plot(n, accu)
    plt.ylim([50, 100])
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy v.s. n')
    plt.savefig(name)
    plt.clf()


  def preprocessData(self):
    data = np.genfromtxt(self.data_f, delimiter = ',')
    # Split data into two halves, one for trainging and the other for test
    # Odd position instaces as training data
    train_data = data[1::2]
    # Even position instaces as test data
    test_data = data[::2]

    # Mapping the last column (class value) from 1 to 1, 0 to -1 by (*2 - 1)
    train_class = train_data[:, -1] * 2 - 1
    test_class = test_data[:, -1] * 2 - 1

    # Scale traing data using standardization except for the last column
    scaler = preprocessing.StandardScaler().fit(train_data[:,:-1])
    train_data = scaler.transform(train_data[:,:-1])
    # Scale test data using mean and stardand deviation from traing data
    test_data = scaler.transform(test_data[:,:-1])

    # Insert class column at the front for SVM LIGHT purposes
    train_data = np.insert(train_data, 0, train_class, axis = 1)
    test_data = np.insert(test_data, 0, test_class, axis = 1)

    return train_data, test_data


  def save2File(self, train_data, test_data):
    # Generate feature indices conforming to SVM LIGHT format
    # Training data
    row = len(train_data)
    column = len(train_data[0])
    trainFile = open('train.data', 'w')
    for i in range(row):
      trainFile.write(str(int(train_data[i][0])) + ' ')
      for j in range(1, column):
        trainFile.write(str(j) + ':' + str(train_data[i][j]) + ' ')

      trainFile.write('\n')
    trainFile.close()

    # Test data
    rows = len(test_data)
    col = len(test_data[0])
    testFile = open('test.data', 'w')
    # For ROC
    testClass = open('test.class', 'w')
    for i in range(rows):
      testFile.write(str(int(test_data[i][0])) + ' ')
      testClass.write(str(int(test_data[i][0])) + '\n')
      for j in range(1, col):
        testFile.write(str(j) + ':' + str(test_data[i][j]) + ' ')

      testFile.write('\n')
    testFile.close()
    testClass.close()


  def plotROC(self):
    targetScore = np.fromfile('output', dtype = float, count = -1, sep = '\n')
    targetClass = np.fromfile('test.class', dtype = int, count = -1, sep = '\n')

    fpr, tpr, thresholds = metrics.roc_curve(targetClass, targetScore)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Test Data')
    plt.savefig('roc_curve.pdf')
    plt.clf()


  def featureSelector(self, i, train_data, test_data):
    # Using weights calculated by the pearl script provided by SVM LIGHT
    wts = {}
    with open('weights') as f:
      for line in f:
        (key, value) = line.split(':')
        wts[int(key)] = abs(float(value))

    wtsSorted = sorted(wts.items(), key = operator.itemgetter(1), reverse = True)

    indList = [e[0] for e in wtsSorted[i:]]
    n_train = np.delete(train_data, indList, axis = 1)
    n_test = np.delete(test_data, indList, axis = 1)

    return n_train, n_test


if __name__== "__main__":
  svm = SVM('spambase.data')
  svm.training()
