import cv2
import numpy

cap = cv2.VideoCapture(0)

lower_blue = numpy.array([110,20,50])
upper_blue = numpy.array([200,255,155])

lower_green = numpy.array([50, 50, 120])
upper_green = numpy.array([255, 155, 155])

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, lower_green, upper_green) # I have the Green threshold image.
    kernel = numpy.ones((5,5),numpy.uint8)
    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = blue_mask + green_mask

    blur = cv2.GaussianBlur(mask,(5,5),21)
    dilated_mask = dilation = cv2.dilate(blur,kernel,iterations = 1)

    cv2.threshold(dilated_mask,127,255,cv2.THRESH_BINARY)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= dilated_mask)


    cv2.imshow('frame',frame)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()