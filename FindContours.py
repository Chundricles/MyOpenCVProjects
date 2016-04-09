__author__ = 'Neil'


import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while(True):
    _, frame = cap.read()

    blur = cv2.GaussianBlur(frame,(5,5),21)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,127,255,0)
    _,contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

    cv2.imshow('frame',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()