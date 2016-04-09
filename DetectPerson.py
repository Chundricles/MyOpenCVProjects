__author__ = 'Neil'
import cv2
import time

def distFromCent(x,y,w,h,centX,centY):
    xPos = x+w/2
    yPos = y+h/2
    xDist = ((centX-xPos)**2)**(0.5)
    yDist = ((centY-yPos)**2)**(0.5)
    dist = ((xDist)**2+(yDist)**2)**(0.5)
    return dist

def xDirection(x,w,centX):
    xPos = x+w/2
    if xPos > centX + 50:
        return "RIGHT"
    if xPos < centX - 50:
        return "LEFT"
    else:
        return "CENTERED"

def yDirection(y,h,centY):
    yPos = y+h/2
    if yPos > centY + 50:
        return "UP"
    if yPos < centY - 50:
        return "DOWN"
    else:
        return "CENTERED"

#set up camera and face cascade.  captures image
cap = cv2.VideoCapture(0)

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


eye = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_eye.xml')
face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
profile1 = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_profileface.xml')
upper = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_upperbody.xml')
full = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_fullbody.xml')

_, frame = cap.read()


#find height, width of image, center x,y positions
height = 480
width = 640
centX = width/2
centY = height/2
fps = 0
count = 0
lastDetected = False


t1 = time.time()

while True:

    dist = 10000
    cX = 0
    cY = 0
    cW = 0
    cH = 0

    skip = False
    foundSomething = False

    #captures first frame and converts to gray
    _, frame = cap.read()

    print frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if  lastDetected == False:
        detect = face.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in detect:
            cv2.rectangle(frame,(x,y),(x+h,y+w),(50,250,250),2)
            foundSomething = True
            print "Default Face"
            thisDist = distFromCent(x,y,w,h,centX,centY)
            if thisDist < dist:
                cX = x
                cY = y
                cW = w
                cH = h
                dist = thisDist

    lastDetected = False

    if count == 0 and foundSomething == False:
        print "searching for eyes"
        detections = eye.detectMultiScale(gray,1.3,5)
        count = count + 1
        skip = True
        for (x,y,w,h) in detections:
            cv2.rectangle(frame,(x,y),(x+h,y+w),(50,250,250),2)
            foundSomething = True
            print "eyes"
            thisDist = distFromCent(x,y,w,h,centX,centY)
            if thisDist < dist:
                cX = x
                cY = y
                cW = w
                cH = h
                dist = thisDist

    if count == 4 and foundSomething == False:
                print "searching for full body"
                detect6 = full.detectMultiScale(gray,1.3,5)
                count = 0
                for (x,y,w,h) in detect6:
                    nx = width - x
                    cv2.rectangle(frame,(nx-w,y),(nx,y+h),(50,250,250),2)
                    foundSomething = True
                    print "Full body"
                    count = 3
                    thisDist = distFromCent(nx,y,w,h,centX,centY)
                    lastDetected = True
                    if thisDist < dist:
                        cX = nx-w
                        cY = y
                        cW = w
                        cH = h
                        dist = thisDist

    if count == 4 and foundSomething == False:
            print "searching for upper body"
            detect5 = upper.detectMultiScale(gray,1.3,5)
            count = count+1
            for (x,y,w,h) in detect5:
                nx = width - x
                cv2.rectangle(frame,(nx-w,y),(nx,y+h),(50,250,250),2)
                foundSomething = True
                print "Upper body"
                count = 3
                thisDist = distFromCent(nx,y,w,h,centX,centY)
                lastDetected = True
                if thisDist < dist:
                    cX = nx-w
                    cY = y
                    cW = w
                    cH = h
                    dist = thisDist

    if count == 3 and foundSomething == False:
        print "searching for HOG person detection"
        detect4, w = hog.detectMultiScale(frame, winStride=(16,16), padding=(32,32), scale=1.05)
        count = count+1
        for (x,y,w,h) in detect4:
            nx = width - x
            cv2.rectangle(frame,(nx-w,y),(nx,y+h),(50,250,250),2)
            foundSomething = True
            print "HOG"
            count = 3
            thisDist = distFromCent(nx,y,w,h,centX,centY)
            lastDetected = True
            if thisDist < dist:
                cX = nx-w
                cY = y
                cW = w
                cH = h
                dist = thisDist

    if count == 2 and foundSomething == False:
        print "searching for right profile"
        detect2 = profile1.detectMultiScale(gray,1.3,5)
        count = count + 1
        for (x,y,w,h) in detect2:
            cv2.rectangle(frame,(x,y),(x+h,y+w),(50,250,250),2)
            foundSomething = True
            print "Right Profile"
            count = 2
            thisDist = distFromCent(x,y,w,h,centX,centY)
            lastDetected = True
            if thisDist < dist:
                cX = x
                cY = y
                cW = w
                cH = h
                dist = thisDist

    if count == 1 and foundSomething == False and skip == False:
        print "searching for alt face profile"
        gray1 = cv2.flip(gray,dst = None, flipCode = 1 )
        detect3 = profile1.detectMultiScale(gray1,1.3,5)
        count = 2
        for (x,y,w,h) in detect3:
            nx = width - x
            cv2.rectangle(frame,(nx-w,y),(nx,y+h),(50,250,250),2)
            foundSomething = True
            print "Left Profile"
            count = 1
            thisDist = distFromCent(nx,y,w,h,centX,centY)
            lastDetected = True
            if thisDist < dist:
                cX = nx-w
                cY = y
                cW = w
                cH = h
                dist = thisDist

    if cX != 0 and cH !=0 :
        text1 = xDirection(cX,cW,centX)
        text2 = yDirection(cY,cH,centY)
        cv2.putText(frame,text1 + " "+ text2,(0,30), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,50,250),2)


    #set closest detection to red rectangle, show image, show text
    cv2.rectangle(frame,(cX,cY),(cX+cW,cY+cH),(0,0,255),2)
    cv2.line(frame,(centX-50,centY-50),(centX-50,centY+50),(255,255,0),1)
    cv2.line(frame,(centX+50,centY-50),(centX+50,centY+50),(255,255,0),1)
    cv2.line(frame,(centX,0),(centX,height),(0,255,0),1)
    cv2.line(frame,(0,centY),(width,centY),(0,255,0),1)
    cv2.line(frame,(centX-50,centY-50),(centX+50,centY-50),(255,255,0),1)
    cv2.line(frame,(centX-50,centY+50),(centX+50,centY+50),(255,255,0),1)
    cv2.putText(frame,'Press ESC to exit',(0,height-20), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,50,250),2)
    cv2.putText(frame,'FPS : '+str(fps),(0,height-50), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,50,250),2)
    cv2.imshow('frame',frame)

    #checks for escape key press, if so exits program
    if cv2.waitKey(1) & 0xFF == 27:
        break

    cycleTime = time.time()-t1
    print cycleTime
    fps = 1/(cycleTime)
    t1 = time.time()

print "Goodbye"
cap.release()
cv2.destroyAllWindows()