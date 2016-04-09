import cv2
import numpy as np
import time
import scipy as sp

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    try :
        for m,n in matches:

            if m.distance < 0.75*n.distance:
                # Get the matching keypoints for each of the images
                img1_idx = m.queryIdx
                img2_idx = n.trainIdx

                # x - columns
                # y - rows
                (x1,y1) = kp1[img1_idx].pt
                (x2,y2) = kp2[img2_idx].pt

                # Draw a small circle at both co-ordinates
                # radius 4
                # colour blue
                # thickness = 1
                cv2.circle(out, (int(x1),int(y1)), 4, (255, 255, 0), 2)
                cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 255, 0), 2)

                # Draw a line in between the two points
                # thickness = 1
                # colour blue
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


        # Show the image
        #cv2.imshow('Matched Features', out)
        #cv2.waitKey(0)
        #cv2.destroyWindow('Matched Features')

        # Also return the image if you'd like a copy
        return out
    except ValueError:
        print 'Value Error'
        return out

img1 = cv2.imread('C:/Users/Neil/Pictures/SamplesUsedForOpenCV/parkingspot.png',0)

cap = cv2.VideoCapture(0)
_, frame = cap.read()


sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
bf = cv2.BFMatcher()

fps = 0

while(True):

    t1 = time.time()

    #captures first frame and converts to gray
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = sift.detectAndCompute(gray,None)
    if des2 != None :
        matches = bf.knnMatch(des1,des2, k=2)


    #matches = []
    #if des1 != None:
    #    for des in des1:
    #        if des2 != None:
    #            for desc in des2:
    #                matches_img = bf.knnMatch(desc,k=2)
    #                for match in matches_img:
    #                    matches.add(match)

    if cv2.waitKey(1) & 0xFF == 27:
            break

    img3 = drawMatches(img1,kp1,gray,kp2,matches)


    cv2.putText(img3,'FPS : '+str(fps),(0,430), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,50,250),2)
    cv2.imshow('frame',img3)
    fps =1/(time.time()-t1)

