__author__ = 'Neil'


import cv2
import time

cap = cv2.VideoCapture(0)

_, frame = cap.read()

height = 480
width = 640

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')

print "Starting Loop"

while(True):
    t1 = time.time()

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, w = hog.detectMultiScale(frame, winStride=(16,16), padding=(32,32), scale=1.05)


    for (x,y,w,h) in found:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

    #if found == ():
    #   detections = face.detectMultiScale(gray,1.3,5)
    #    for (x,y,w,h) in detections:
    #       cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

    cv2.putText(frame,'Press ESC to exit',(0,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.imshow('frame',frame)

    print time.time() - t1

    if cv2.waitKey(1) & 0xFF == 27:
        break




cap.release()
cv2.destroyAllWindows()