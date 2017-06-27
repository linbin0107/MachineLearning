#!/usr/bin/env python

'''This program implements the naive Bayes classification algorithm
   and logistic regression.'''

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class BayesLearning:
  def __init__(self, data_f):
    self.data_f = data_f


  # Create probabilistic model for train data
  def createModel(self, train_data, train_class):
    #split the training data into spam and non-spam class
    pos_cnt = np.count_nonzero(train_class)
    #print(pos_cnt)
    #print(len(train_class))
    neg_cnt = len(train_class) - pos_cnt
    #print(float(pos_cnt) / len(train_class))
    pos_data = np.empty((pos_cnt, len(train_data[0])))
    neg_data = np.empty((neg_cnt, len(train_data[0])))

    pos_ind = 0
    neg_ind = 0

    for i in range(len(train_class)):
      if train_class[i] == 1.0:
        pos_data[pos_ind] = train_data[i]
        pos_ind += 1
      else:
        neg_data[neg_ind] = train_data[i]
        neg_ind += 1

    # Compute the mean and standard deviation for training data
    pos_mean, pos_stdev = self.compute(pos_data)
    neg_mean, neg_stdev = self.compute(neg_data)

    return pos_mean, pos_stdev, pos_cnt, neg_mean, neg_stdev, neg_cnt

  def compute(self, data):
    mean = np.mean(data, axis = 0)
    stdev = np.std(data, axis = 0)
    # Replace zeros with 0.0001 in stdev
    stdev[stdev == 0.0] = 0.0001

    return mean, stdev


  def runNaiveBayes(self, pos_mean, pos_stdev, pos_cnt, neg_mean, neg_stdev, \
               neg_cnt, test_data, test_class):
    row = len(test_data)
    col = len(test_data[0])
    pos_p = np.empty(col)
    neg_p = np.empty(col)
    res = np.empty(row)

    pp = float(pos_cnt) / (pos_cnt + neg_cnt)
    nnp = float(neg_cnt) / (pos_cnt + neg_cnt)

    # Compute P(x|class) of both classes for test data
    for i in range(row):
      pos_p = -((test_data[i] - pos_mean) ** 2) / (2.0 * (pos_stdev ** 2))
      pos_p = np.exp(pos_p)
      pos_p = (1 / (np.sqrt(2 * np.pi) * pos_stdev)) * pos_p
      neg_p = -((test_data[i] - neg_mean) ** 2) / (2.0 * (neg_stdev ** 2))
      neg_p = np.exp(neg_p)
      neg_p = (1.0 / (np.sqrt(2 * np.pi) * neg_stdev)) * neg_p

      pos = np.sum(np.log(pos_p)) + np.log(pp)
      neg = np.sum(np.log(neg_p)) + np.log(nnp)

      res[i] = 1 if pos > neg else 0

    # Compute FP, FN, TP, and TN
    fp, fn, tp, tn = self.analyseRes(res, test_class)

    accuracy = float(tp + tn) / row
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)

    self.createConMatrix(fp, fn, tp, tn, 'BayesConMatrix')


  # Classification with Logitstic Regression using scikit_learn library
  def logisticRegression(self, train_data, train_class, test_data, test_class):
    model = LogisticRegression()
    model = model.fit(train_data, train_class)
    test_res = model.predict(test_data)

    fp, fn, tp, tn = self.analyseRes(test_res, test_class)

    accuracy = float(tp + tn) / len(test_class)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)

    self.createConMatrix(fp, fn, tp, tn, 'LRConMatrix')


  def createConMatrix(self, fp, fn, tp, tn, name):
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.axis('off')
    axs.xaxis.set_visible(False)
    axs.yaxis.set_visible(False)

    data = [[tp, fn],[fp, tn]]
    labels = ['positive', 'negative']

    tb = plt.table(cellText = data, colWidths = [0.2] * 3, loc = (0.1, 0), \
         cellLoc = 'center', rowLabels = labels, colLabels = labels)
    tb.set_fontsize(10)
    tb.scale(1, 2.5)

    plt.title('Confusion Matrix')
    plt.text(0, 0.5, 'Actual Class')
    plt.text(0.35, 0.75, 'Predicted Class')

    fig.savefig(name)


  def analyseRes(self, res, test_class):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    size = len(res)

    for i in range(size):
      if res[i] == 1 and test_class[i] == 1:
        tp += 1
      elif res[i] == 1 and test_class[i] == 0:
        fp += 1
      elif res[i] == 0 and test_class[i] == 0:
        tn += 1
      else:
        fn += 1

    return fp, fn, tp, tn


  def preprocessData(self):
    data = np.genfromtxt(self.data_f, delimiter = ',')
    # Split data into two halves, one for trainging and the other for test
    # Odd position instaces as training data
    train_data = data[1::2]
    # Even position instaces as test data
    test_data = data[::2]

    # Extract the last column (the class value)
    train_class = train_data[:, -1]
    test_class = test_data[:, -1]

    # Remove the last column from the data
    train_data = np.delete(train_data, -1, axis = 1)
    test_data = np.delete(test_data, -1, axis = 1)

    return train_data, test_data, train_class, test_class


if __name__== "__main__":
  bayes = BayesLearning('spambase.data')
  # Preprocessing data
  train_data, test_data, train_class, test_class = bayes.preprocessData()
  # Create probabilistic model
  pos_mean, pos_stdev, pos_cnt, neg_mean, neg_stdev, neg_cnt = bayes.createModel(train_data, train_class)
  # Run naive Bayes classification
  bayes.runNaiveBayes(pos_mean, pos_stdev, pos_cnt, neg_mean, neg_stdev, neg_cnt, test_data, test_class)
  # Run logistic regression classification
  bayes.logisticRegression(train_data, train_class, test_data, test_class)
