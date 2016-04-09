__author__ = 'Neil'
import cv2
import time
import numpy as np

#set up camera and face cascade.  captures image
cap = cv2.VideoCapture(0)
_, frame = cap.read()


templ = cv2.imread('C:/Users/Neil/Pictures/SamplesUsedForOpenCV/parkingspot.png',0)
template = cv2.Canny(templ,50,200)
w, h = template.shape[::-1]

out = cv2.VideoWriter('ParkingSpotExample.avi',-1,20,(640,480))



while(True):

    t1 = time.time()

    detectX = []

    #captures first frame and converts to gray
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(frame, 150,200)

    cv2.imshow('template',template)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(edged,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.08
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        detectX.append(pt[0]+w/2)

    if len(detectX)>0:
        avgX = sum(detectX)/len(detectX)
        cv2.line(frame,(avgX,0),(avgX,480),(255,0,255),2)

    #checks for escape key press, if so exits program
    if cv2.waitKey(1) & 0xFF == 27:
        break


    cv2.imshow('frame',frame)
    cv2.imshow('contours',edged)
    out.write(frame)
    print time.time()-t1


print "Goodbye"
cap.release()
out.release()
cv2.destroyAllWindows()