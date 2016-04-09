__author__ = 'Neil'
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

_, frame = cap.read()

fps = 0

while(True):

    t1 = time.time()

    #captures first frame and converts to gray
    _, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(frame, 150,200)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,100,5)

    if lines != None :
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
            slope = (x2-x1)/(y2-y1)

    cv2.putText(frame,'FPS : '+str(fps),(0,430), cv2.FONT_HERSHEY_TRIPLEX, 1,(255,255,255),2)
    cv2.putText(frame,'Press ESC to exit',(0,460), cv2.FONT_HERSHEY_TRIPLEX, 1,(255,255,255),2)
    cv2.imshow('frame',frame)


    #checks for escape key press, if so exits program
    if cv2.waitKey(1) & 0xFF == 27:
        break

    cycleTime = time.time()-t1
    fps = 1/(cycleTime)

cap.release()
cv2.destroyAllWindows()