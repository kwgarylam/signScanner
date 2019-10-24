import time
import cv2
import numpy as np
import os

#Frame width & Height
w=320
h=240

match = -1
font = cv2.FONT_HERSHEY_SIMPLEX

#Reference Images Display name & Original Name

ReferenceImages = ["left.jpg","right.jpg","reverse.jpg", "start.jpg", "stop.jpg", "magic.jpg"]
ReferenceTitles = ["Turn Left", "Turn Right", "Reverse", "VTC STEM", "Stop", "Magic"]

#define class for References Images
class Symbol:
    def __init__(self):
        self.img = 0
        self.name = 0

#define class instances (6 objects for 6 different images)
symbol= [Symbol() for i in range(6)]

def readRefImages():
        for count in range(6):
                imageRef = cv2.imread(ReferenceImages[count], cv2.COLOR_BGR2GRAY)
                imageRefResized = cv2.resize(imageRef,(int(w/2),int(h/2)),interpolation = cv2.INTER_AREA) 

                symbol[count].img = resize_and_threshold_warped(imageRefResized)
                symbol[count].name = ReferenceTitles[count] 
                 
                #print symbol[count].img
                #print symbol[count].name
                #cv2.imshow(symbol[count].name,symbol[count].img);
                
def order_points(pts):
        #Ref. https://gist.github.com/nikgens/1a129d620978a4abc6a9a30f5f66e0d3
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(pts)

        maxWidth = w/2
        maxHeight = h/2

        # construct our destination points which will be used to # map the screen to a top-down, "birds eye" view
        dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])

        # calculate the perspective transform matrix and warp
        # # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        #warped = image

        # return the warped image
        return warped


def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged


def resize_and_threshold_warped(image):
        #Resize the corrected image to proper size & convert it to grayscale
        #warped_new =  cv2.resize(image,(w/2, h/2))
        warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Smoothing Out Image
        blur = cv2.GaussianBlur(warped_new_gray,(5,5),0)

        #Calculate the maximum pixel and minimum pixel value & compute threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        threshold = (min_val + max_val)/2

        #Threshold the image
        ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)

        #return the thresholded image
        return warped_processed

def checkImage(OriginalFrame, contours, vertex, minSquareArea, minDiff):
        for cnt in contours:
                # Approximates a polygonal curve(s) with the specified precision.
                approxVertex = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

                #If the detected objects have 4 vertex and larger than a minimal size
                if len(approxVertex)==vertex:
                        #Calculate the area of the square
                        area = cv2.contourArea(approxVertex)
                        #print area

                        if area > minSquareArea:

                                # Draw the rectangle on the square
                                cv2.drawContours(OriginalFrame,[approxVertex],0,(0,0,255),3)

                                warped = four_point_transform(OriginalFrame, approxVertex.reshape(4, 2))

                                #Histogram Equalization, converting image from grayscale to binary 
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                                warpedDilated= cv2.dilate(warped,kernel,iterations = 1)
                                warped_eq = resize_and_threshold_warped(warpedDilated)

                                for i in range(6):
                                        diffImg = cv2.bitwise_xor(warped_eq, symbol[i].img)
                                        diff = cv2.countNonZero(diffImg)
                                        #print diff

                                        if diff < minDiff:
                                                match = i
                                                
                                                #Accessing Values in Tuples, get the top left conner of the vertex
                                                cv2.putText(OriginalFrame,symbol[match].name, tuple(approxVertex.reshape(4,2)[0]), font, 1, (255,0,255), 2, cv2.LINE_AA)
                                                cv2.imshow("Matching Operation", diffImg)

                                                print ("Matched Image: "+symbol[match].name)          
                                                print ("The difference between Captured Image and Database")
                                                print (diff)
                                                break
