#!/usr/bin/env python3
from os import isatty
from dnn import *
import pandas as pd
import numpy as np
import cv2

drawing = False
ix, iy = -1, -1
train = False
dnn = DNN()

def draw(event, x, y, flags, params):
    global drawing, ix, iy 
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 2, (255, 255, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.zeros((28, 28), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

if train:
    dnn.W1, dnn.b1, dnn.W2, dnn.b2 = dnn.grad_desc(X_train, Y_train, 0.001, 5000)
    dnn.save_weight()

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

img_temp = cv2.resize(img, (28, 28),0, 0, cv2.INTER_NEAREST)
img_temp = img_temp[:, :]
img_test = img_temp.flatten().T
img_norm = img_test/255
img_norm = img_norm.reshape(-1, 1)

dnn.test_pred(img_norm)

current_image = img_norm.reshape((28, 28)) * 255
current_image = cv2.resize(current_image, (280, 280),0, 0, cv2.INTER_NEAREST)
cv2.imshow('image', current_image)
cv2.waitKey(0)