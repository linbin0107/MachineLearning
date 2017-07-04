#!/usr/bin/env python

'''
Implements K-means clustering using Optdigit dataset
Author: Bin Lin
'''


import numpy as np
import random

class K_means:
  def __init__(self, k, dim, num_of_run):
    self.k = k
    self.dim = dim
    self.num_of_run = num_of_run
    self.train_data = []
    self.test_data = []
    self.train_class = []
    self.test_class = []


  def mainCluster(self):
    # Save resulting data for each run
    print('The number of clusters: %d\n' % (self.k))
    kmeans_run = []

    for i in range(self.num_of_run):
      ranClus = self.genRanCluster()
      labels, trained_clus, blong2clus, amse = self.trainCluster(ranClus)
      kmeans_run.append((labels, trained_clus, blong2clus, amse))

    print('All %d runs are completed.' % (self.num_of_run))

    # Choose the run (out of num_of_run) that yields the smallest amse
    indx = 0
    for i in range(1, len(kmeans_run)):
      if kmeans_run[i][3] < kmeans_run[indx][3]:
        indx = i
    print('The best run was number %d\n' % (indx+int(1)))

    # For this best run, reports the amse, mss, and mean entropy
    print("Avarage mse: ", kmeans_run[indx][-1])
    self.evalCluster(kmeans_run[indx][1], kmeans_run[indx][2])

    # Assign each test instance the class of the closest cluster center.
    # Ties are broken at random
    accu = self.classifyTest(kmeans_run[indx][0], kmeans_run[indx][1])
    print("Accuracy on test data: ", accu)


  def classifyTest(self, labels, clus):
    size = len(self.test_data)

    dist = np.empty((size, self.k))
    pred_class = np.empty(size)
    pred_clus = np.empty(size)

    for i in range(size):
      for j in range(self.k):
        dist[i][j] = np.sum((self.test_data[i] - clus[j]) ** 2)
      pred_clus[i] = np.argmin(dist[i])
      pred_class[i] = labels[int(pred_clus[i])]

    cnt = 0
    for i in range(size):
      if pred_class[i] == self.test_class[i]:
        cnt += 1

    accu = float(cnt) / size

    return accu


  def evalCluster(self, cluster, blong2clus):
    # Compute mean square separation (mss)
    mss = 0
    for i in range(self.k):
      for j in range(self.k):
        if i != j:
          mss += np.sum((cluster[i] - cluster[j]) ** 2)

    mss /= self.k * (self.k - 1) / 2
    print("mss: ", mss)

    # Compute mean entropy
    size = len(self.train_data)
    clus_cnt = np.zeros((self.k, self.k))
    for j in range(size):
      clus_cnt[blong2clus[j]][self.train_class[j]] += 1

    men = 0
    for i in range(self.k):
      num = np.sum(clus_cnt[i])
      if num == 0:
        continue;
      en = 0
      for j in range(self.k):
        if clus_cnt[i][j] == 0:
          continue
        en += clus_cnt[i][j] / num * np.log2(clus_cnt[i][j] / num)
      en = -en
      men += float(num) / size * en

    print("mean entropy: ", men)


  def trainCluster(self, cluster):
    size = len(self.train_data)
    blong2clus = np.empty(size)

    # Train until the centroids do not change
    while True:
      new_clus = np.zeros((self.k, self.dim))
      # Find each data point to the closest cluster centroid
      dist = np.empty((size, self.k))
      for i in range(size):
        for j in range(self.k):
          dist[i][j] = np.sum((self.train_data[i] - cluster[j]) ** 2)
        blong2clus[i] = np.argmin(dist[i])

      # Update the cluster centroids
      mse = np.zeros(self.k)
      clus2copy = []
      for i in range(self.k):
        cnt = 0
        for j in range(size):
          if blong2clus[j] == i:
            new_clus[i] += self.train_data[j]
            mse[i] += np.sum((self.train_data[j] - cluster[i]) ** 2)
            cnt += 1

        if cnt == 0:
          clus2copy.append(i)
        else:
          mse[i] /= cnt
          new_clus[i] /= float(cnt)

      max_mse_indx = np.argmax(mse)
      for indx in clus2copy:
        new_clus[indx] = cluster[max_mse_indx]
        #new_clus[indx] = cluster[indx]

      diff = np.absolute(new_clus - cluster)
      if (diff < 0.01).all():
        amse = np.sum(mse) / self.k
        break

      cluster = new_clus

    # Associate each cluster center with the most frequent class it
    # contains in the training data. If there is a tie for most frequent
    # class, break the tie at random
    labels = np.zeros(self.k)
    for i in range(self.k):
      label = np.zeros(10)
      for j in range(size):
        if blong2clus[j] == i:
          label[self.train_class[j]] += 1

      # Randomly break the tie
      #labels[i] = np.argmax(label)
      labels[i] = np.random.choice(np.flatnonzero(label == label.max()))

    return labels, new_clus, blong2clus, amse


  def genRanCluster(self):
    # Generate k clusters randomly
    cluster = np.empty((self.k, self.dim))
    for i in range(self.k):
      for j in range(self.dim):
        cluster[i][j] = random.randint(0, 16)

    return cluster


  def preprocessingData(self, trainFile, testFile):
    self.train_data = np.genfromtxt(trainFile, delimiter = ',')
    self.test_data = np.genfromtxt(testFile, delimiter = ',')

    self.train_class = self.train_data[:, -1]
    self.test_class = self.test_data[:, -1]

    self.train_data = np.delete(self.train_data, -1, axis = 1)
    self.test_data = np.delete(self.test_data, -1, axis = 1)


if __name__ == "__main__":
  # K = 10
  kmeans_10 = K_means(10, 64, 5)
  kmeans_10.preprocessingData('./optdigits/optdigits.train', \
             './optdigits/optdigits.test')
  kmeans_10.mainCluster()
  # K = 50
  kmeans_50 = K_means(50, 64, 5)
  kmeans_50.preprocessingData('./optdigits/optdigits.train', \
             './optdigits/optdigits.test')
  kmeans_50.mainCluster()
