# ======================================================= #
# Author: VTC STEM Centre/ Gary Lam
# Date: 27/7/2018
# Program Name: RoadSign Game version 1.2
# Description: Program code for sign detection
# ====================================================== #

# import the necessary packages
import time
import cv2
import numpy as np
import os
import signDetectionLib as sign

#Difference Variable
minDiff = 3000
minSquareArea = 2000
vertex = 4

video = cv2.VideoCapture(0)

sign.readRefImages()

print ("Please use the Road Sign to play this demo.")
print ("Please press q to exit")

while True:
        #Load the video frame from camera and convert the frame to gray color
        ret, OriginalFrame = video.read()
        gray = cv2.cvtColor(OriginalFrame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(3,3),0)
        
        # Detecting Edges
        edges = sign.auto_canny(blurred)

        # Contour Detection & checking for squares based on the square area
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(OriginalFrame, contours, -1, (0,255,0), 3)

        # Send the Image for checking
        sign.checkImage(OriginalFrame, contours, vertex, minSquareArea, minDiff)

        # Show Image
        cv2.imshow("Main Frame", OriginalFrame)
        #cv2.imshow("Gray", gray)
        #cv2.imshow("Blurred", blurred)
        #cv2.imshow("Edges", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video.release()
cv2.destroyAllWindows()
