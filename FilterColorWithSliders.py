import cv2
import numpy as np
import time

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')

cv2.createTrackbar('R UPPER','image',0,255,nothing)
cv2.createTrackbar('G UPPER','image',0,255,nothing)
cv2.createTrackbar('B UPPER','image',0,255,nothing)
cv2.createTrackbar('R LOWER','image',0,255,nothing)
cv2.createTrackbar('G LOWER','image',0,255,nothing)
cv2.createTrackbar('B LOWER','image',0,255,nothing)

time.sleep(1)

while(1):

    Rup = cv2.getTrackbarPos('R UPPER','image')
    Gup = cv2.getTrackbarPos('G UPPER','image')
    Bup = cv2.getTrackbarPos('B UPPER','image')
    Rlow = cv2.getTrackbarPos('R LOWER','image')
    Glow = cv2.getTrackbarPos('G LOWER','image')
    Blow = cv2.getTrackbarPos('B LOWER','image')

    lower_blue = np.array([Blow,Glow,Rlow])
    upper_blue = np.array([Bup,Gup,Rup])

    # Take each frame
    _, frame = cap.read()

    img = cv2.GaussianBlur(frame,(5,5),5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = blue_mask

    blur = cv2.GaussianBlur(mask,(5,5),21)
    cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= blur)

    cv2.imshow('image',frame)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print "Blue Upper : "+ str(Bup)
print "Blue Lower : "+ str(Blow)
print "Green Upper : "+ str(Gup)
print "Green Lower : "+ str(Glow)
print "Red Upper : "+ str(Rup)
print "Red Lower : "+ str(Rlow)