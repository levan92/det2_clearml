import cv2
import numpy as np

frame = np.zeros((1080, 1920))

l = 100
t = 100
w = 256
h = 256
r = l + w
b = t + h

cv2.rectangle(frame, (l,t), (r,b), (255,255,0), 1)

cv2.imshow('', frame)
cv2.waitKey(0)