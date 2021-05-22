#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import fetch_openml
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
# x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

data = pd.read_csv('traincsv.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

class DNN():
    def __init__(self):
        if os.path.isfile("weights/w1.csv"):
            self.W1 = np.array(pd.read_csv('weights/w1.csv'))
        else:
            self.W1 = np.random.rand(10, 784) - 0.5

        if os.path.isfile('weights/b1.csv'):
            self.b1 = np.array(pd.read_csv('weights/b1.csv'))
        else:
            self.b1 = np.random.rand(10, 1) - 0.5
        if os.path.isfile('weights/w2.csv'):
            self.W2 = np.array(pd.read_csv('weights/w2.csv'))
        else:   
            self.W2 = np.random.rand(10, 10) - 0.5
        if os.path.isfile('weights/b2.csv'):
            self.b2 = np.array(pd.read_csv('weights/b2.csv'))
        else:
            self.b2 = np.random.rand(10, 1) - 0.5

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def feedforward(self, X):
        Z0 = self.W1.dot(X) + self.b1
        A0 = self.relu(Z0)
        Z1 = self.W2.dot(A0) + self.b2
        A1 = self.softmax(Z1)

        return Z0, A0, Z1, A1

    def relu(self,X, derivative=False):
        if derivative:
            return X>0
        return np.maximum(X, 0)

    def backProp(self, X, Z0, A0, Z1, A1, y_train):
        y_train = self.one_hot(y_train)

        error = (A1 - y_train)
        change_w2 = np.dot(error, A0.T)/m
        error = np.dot(self.W2.T, error)*self.relu(Z0, True)
        change_w1 = np.dot(error, X.T)/m
        change_b2 = 1/m*np.sum(change_w2)
        change_b1 = 1/m*np.sum(change_w1)
        return change_w1, change_b1, change_w2, change_b2

    def update_params(self, dw1, db1, dw2, db2, l_r):
        self.W1 -= l_r*dw1
        self.b1 -= l_r*db1
        self.W2 -= l_r*dw2
        self.b2 -= l_r*db2

    def grad_desc(self, X, Y, l_r, iter):
        for i in range(iter):
            Z0, A0, Z1, A1 = self.feedforward(X)
            dw1, db1, dw2, db2 = self.backProp(X, Z0, A0, Z1, A1, Y)
            self.update_params(dw1, db1, dw2, db2, l_r)
            if i%10==0:
                predictions = self.get_pred(A1)
                print('Iter:', i)
                print(self.get_accuracy(predictions, Y))
        return self.W1, self.b1, self.W2, self.b2


    def get_pred(self, A1):
        return np.argmax(A1, 0)
    
    def get_accuracy(self,predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size


    def fit(self, X):
        _, _, _, A2 = self.feedforward(X)
        pred = self.get_pred(A2)
        return pred

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def test_pred(self,img):
        prediction = self.fit(img)
        print("Prediction: ", prediction)

    

    
    def save_weights(self):
        w1 = pd.DataFrame(self.W1)
        w2 = pd.DataFrame(self.W2)
        b1 = pd.DataFrame(self.b1)
        b2 = pd.DataFrame(self.b2)

        w1.to_csv('weights/w1.csv', index=False)
        w2.to_csv('weights/w2.csv', index=False)
        b1.to_csv('weights/b1.csv', index=False)
        b2.to_csv('weights/b2.csv', index=False)



# if __name__ == '__main__':
#     dnn = DNN()
#     train = False
#     if train:
#         dnn.W1, dnn.b1, dnn.W2, dnn.b2 = dnn.grad_desc(X_train, Y_train, 0.05, 1000)
#         dnn.save_weights()

#     else:
#         img = cv2.imreadZZ
#         dnn.test_pred(5)


