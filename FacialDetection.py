__author__ = 'Neil'
import cv2
import time

#check to see if selection is closest
def distFromCent(x,y,w,h,centX,centY):
    xPos = x+w/2
    yPos = y+h/2
    xDist = ((centX-xPos)**2)**(0.5)
    yDist = ((centY-yPos)**2)**(0.5)
    dist = ((xDist)**2+(yDist)**2)**(0.5)
    return dist


#set up camera and face cascade.  captures image
cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
#face = cv2.CascadeClassifier('C:\opencv\sources\data\lbpcascades\lbpcascade_frontalface.xml')
_, frame = cap.read()


#find height, width of image, center x,y positions
height,width,_ =  frame.shape
centX = height/2
centY = width/2

#sets up initial values
dist = 10000
cX = 0
cY = 0
cW = 0
cH = 0
closest = True


while(True):

    t1 = time.time()

    #captures first frame and converts to gray
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detects faces using cascade
    detections = face.detectMultiScale(gray,1.3,5)

    #loops for all detected faces, puts rectangles around them, finds closest to center
    for (x,y,w,h) in detections:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)
        thisDist = distFromCent(x,y,w,h,centX,centY)
        if thisDist < dist:
            cX = x
            cY = y
            cW = w
            cH = h
            dist = thisDist

    #set closest detection to red rectangle, show image, show text
    cv2.putText(frame,'Press ESC to exit',(0,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.rectangle(frame,(cX,cY),(cX+cW,cY+cH),(0,0,255),2)
    cv2.imshow('frame',frame)

    #checks for escape key press, if so exits program
    if cv2.waitKey(1) & 0xFF == 27:
        break

    #reset intial values
    dist = 10000
    cX = 0
    cY = 0
    cW = 0
    cH = 0

    print time.time()-t1

cap.release()
cv2.destroyAllWindows()