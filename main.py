#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

from os import write
import numpy as np
from numpy.compat import py3k
from numpy.lib.arraypad import _set_wrap_both 
import coms
from forward import s3
import os
from coordinates import wcoordinates, rcoordinates
from cgi import test
from typing import NoReturn
from Orange.preprocess import discretize
from Orange.statistics.distribution import Discrete
import cv2
import imutils
import numpy as np
from Orange.data import domain
from orangewidget.gui import label
from sklearn.metrics import pairwise
from sklearn import linear_model
import os
import Orange
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from orangecontrib.imageanalytics.import_images import ImportImages
from Orange import classification
from Orange import modelling
from Orange import evaluation
from Orange import preprocess
from Orange.evaluation import testing

# Begin Arduino programming via sketch.
# print("Load Arduino Sketch \"setupsketch.ino\" ? (y/n)")
# keypress = input()
# if keypress == ord("y"):
#     print("Loading sketch...")
#     coms.upload()

#Initialize serial
try:
    com = coms.dueserial()
except:
    print("\n Unable to connect to serial device.")

base = bytes(90)
shoulder = bytes(45)
elbow = bytes(180)
wrist = bytes(180)
wrot = bytes(90)
grip = bytes(10)
step = bytes(10)


# Serial write packet.
data_out = [step, base, shoulder, elbow ,wrist, wrot, grip]

# Robot Dimensions.
# 1st arm from shoulder, forecep, wrist with gripper length.
a = np.array([12.5, 12.5, 18.5])

# Physical Constraints
# 0 is shoulder
# 1 is elbow
# 2 is Wrist V
# 4 is Wrist R
# 5 is Grabber

base_min = 0
base_max = 180
shoulder_min = 15
shoulder_max = 165
elbow_min = 0
elbow_max = 180
wristv_min = 0
wristv_max = 180
wristr_min = 0
wristr_max = 180
gripper_min = 10
gripper_max = 73

constraint = np.array([  [shoulder_min, shoulder_max],             
                            [elbow_min, elbow_max],                 
                            [wristv_min, wristv_max],           
                            [wristr_min,wristr_max],        
                            [gripper_min, gripper_max]  ])

# Float values 0 to 180
theta = np.arange(start=0, stop=181, step=1, dtype=int)
euclidean = np.empty((45,45,3))



wcoordinates(theta, euclidean)

euclidean = rcoordinates()

def setout(prediction):
    

# global variables.widgets.evaluate
bg = None

#counter
def foo():
    foo.counter += 1
    print ("Counter is %d" % foo.counter)
    return foo.counter
foo.counter = 0

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=50, thresholdtype=cv2.THRESH_BINARY):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 256, thresholdtype)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":

    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 450, 225, 690

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # Construct Training image path
    basepath = os.path.dirname(os.path.realpath(__file__))
    imagefolder = "gestures"
    testfolder = "test"
    path = (basepath + "\\" + imagefolder)
    testpath = (basepath + "\\" + testfolder)
    print(path)

    # Test image path
    print(testpath)

    # Initialize Image Import
    imimp = ImportImages()

    # Import Training images
    imdata, err = imimp(path)

    # Check images properly imported.
    # print(imdata.domain)
    # print(imdata)
    # print(type(imdata))
    # print(testdata)

    # Initialize Image Embedder
    imemb = ImageEmbedder(model="squeezenet")

    # Embed Training images
    imembdata, skippedim, numskippedim = imemb(imdata, col="image")

    # print(imembdata)
    # print(skippedim)
    # print(numskippedim)

    # Initialize learner
    # learner = classification.naive_bayes.NaiveBayesLearner()
    # learner = classification.TreeLearner
    learner = classification.KNNLearner()

    # Train learner for model
    lmodel = learner(imembdata)
    
    # Set object for getting class values from data based on prediction
    classval = imembdata.domain.class_var.values

    # Set Thresholding type flag
    threshtype = cv2.THRESH_BINARY
    thresh = 30

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()
        clone2 = clone.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(image=gray, threshold=thresh, thresholdtype=threshtype)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                #fill Contours
                cv2.fillPoly(clone, [segmented + (right, top)], (255, 255, 255))
                cv2.fillPoly(thresholded, [segmented + (right, top)], (255, 255, 255))

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.drawContours(thresholded, [segmented + (right, top)], -1, (0, 0, 255))

                # save segmented frame to testpath
                cv2.imwrite(testpath + "\\" + "test.jpg", thresholded)

                #testing timing qualitatively
                # print("imwrite done")

                # Import Test image
                testdata, err = imimp(testpath)

                #testing timing qualitatively
                #print("Import done")

                # Embed test image
                testemb, skippedim, numskippedim = imemb(testdata, col="image")

                #testing timing qualitatively
                # print("Embed done")

                # Make prediction using learner
                prediction = lmodel(testemb)
                
                cv2.putText(clone, str(classval[int(prediction)]), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

                com.write_serial(data_out)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user~
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        elif(keypress == ord("r")):
            num_frames = 0
        elif(keypress == ord("i")):
            if(threshtype == cv2.THRESH_BINARY):
                threshtype == cv2.THRESH_BINARY_INV
            else:
                threshtype = cv2.THRESH_BINARY
        elif(keypress == ord("u")):
            if(thresh < 256):
                thresh += 2
                print(thresh)
            else:
                thresh = thresh
                print(thresh)
        elif(keypress == ord("d")):
            if(thresh >0):
                thresh -= 2
                print(thresh)
            else:
                thresh = thresh
                print(thresh)
        else:
            if keypress == ord("c"):
                #cv2.imshow("gray", thresholded)
                filepath = path + "\\" + "recorded images"
                status = cv2.imwrite(filepath + "gestureframe%d.jpg" % foo(), thresholded)
                if status == True:
                    print("entered img save")
                else:
                    print("Save Failure")

    # free up memory
    camera.release()
    cv2.destroyAllWindows()
