#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

from deap.tools.support import HallOfFame
from numpy.core.defchararray import add, array, multiply
from numpy.core.function_base import linspace
from numpy.core.numeric import full
import numpy as np 
import coms
from forward import s3,s2,s1
import math
from math import exp, radians as rad, sqrt
from math import degrees as deg
from deap import creator
from deap import gp 
from deap import base
from deap import tools
from deap import algorithms
import operator
import random
import operator

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

# Serial write packet.
data_out = [10, 90, 45, 180 ,180, 90, 10]

# Robot Dimensions.
# 1st arm from shoulder, forecep, wrist with gripper length.
a = np.array([12.5, 12.5, 18.5])

# Physical Constraints
# 0 is shoulder
# 1 is elbow
# 2 is Wrist V
# 4 is Wrist R
# 5 is Grabber

base_min = 0.0
base_max = 180.0
shoulder_min = 15.0
shoulder_max = 165.0
elbow_min = 0.0
elbow_max = 180.0
wristv_min = 0.0
wristv_max = 180.0
wristr_min = 0.0
wristr_max = 180.0
gripper_min = 10.0
gripper_max = 73.0

constraint = np.array([  [shoulder_min, shoulder_max],             
                            [elbow_min, elbow_max],                 
                            [wristv_min, wristv_max],           
                            [wristr_min,wristr_max],        
                            [gripper_min, gripper_max]  ])

theta = np.zeros((np.linspace(0,180,10).shape[0], 3), dtype=float)
oneeighty = np.linspace(0, 180, 10)

for x in range(theta.shape[0]):
    for y in range(theta.shape[1]):
        theta[x][theta.shape[1]-1] = oneeighty[x]
        theta[x][theta.shape[1]-2] = oneeighty[theta.shape[0]-x-1]
        theta[x][theta.shape[1]-3] = oneeighty[x]

print(theta)

# Extract single dimension values for x y values from transform
s3 = np.array([s3(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])
s2 = np.array([s2(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])
s1 = np.array([s1(theta[i][0], theta[i][1],theta[i][2]) for i in range(theta.shape[0])])

s = np.array([s1, s2, s3])

# Transform degrees to radians    
for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
        theta[i][j] = rad(theta[i][j])

