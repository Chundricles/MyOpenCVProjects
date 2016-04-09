__author__ = 'Neil'

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

time.sleep(1)

lower_blue = np.array([0,0,127])
upper_blue = np.array([204,51,255])

#lower_green = np.array([0,0,127])
#upper_green = np.array([204,51,255])

while(1):

    # Take each frame
    _, frame = cap.read()
    blur = cv2.GaussianBlur(frame,(5,5),21)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #green_mask = cv2.inRange(hsv, lower_green, upper_green) # I have the Green threshold image.

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = blue_mask #+ green_mask

    dilate = cv2.dilate(mask,(5,5),iterations = 13)

    res = cv2.bitwise_and(frame,frame, mask= dilate)
    median = cv2.medianBlur(res,5)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    #notThresh,thresh = cv2.threshold(gray,50,255,0)

    contours,_ = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours :
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

        if w*h>10000:
            cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow('frame',frame)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()