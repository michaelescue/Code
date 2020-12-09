#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference


import queue
from numpy.core.defchararray import array
from numpy.core.numeric import full
import time
from pynput import mouse
import numpy as np 
from queue import Queue

from tensorflow.python.ops.gen_math_ops import asin
import coms
import tensorflow as tf

# Begin Arduino programming via sketch.
print("Load Arduino Sketch \"setupsketch.ino\" ? (y/n)")
keypress = input()
if keypress == ord("y"):
    print("Loading sketch...")
    coms.upload()

#Initialize serial
try:
    com = coms.dueserial()
except:
    print("Unable to connect to serial device.\n")

# Lengths
# 1st arm from shoulder, forecep, wrist with gripper length.
a = tf.Variable([12.5, 12.5, 18.5])

# X axis offset
d = tf.Variable([0.0, 0.0, 0.0, 0.0])

# Joint values
theta = tf.Variable([45.0, 180.0, 180.0, 90.0])

# Axis Twist angles
alpha = tf.Variable([0.0, 0.0, 0.0, -90.0])

# Serial write packet.
data_out = [10, 90, 45, 180 ,180, 90, 10]

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

constraint = tf.constant([  [shoulder_min, shoulder_max],             
                            [elbow_min, elbow_max],                 
                            [wristv_min, wristv_max],           
                            [wristr_min,wristr_max],        
                            [gripper_min, gripper_max]  ])

print("Constraints:\n", constraint)

# DH Matrix 
# thetai: A rotation angle between two links, about the z-axis
# di: The distance(offset) on the z-axis, between links(prismatic)
# ai: The length of each common normal (Joint offset)
# alphai: The twist angle between two successive z-axes (Joint twist/revolute)
# Individual rows are of type "tf.Tensor"
# [joint] = 
#                   [alphai, ai,   di,      thetai]
DH = tf.Variable([  [0.0,    0.0,    d[0],    theta[0]],
                    [0.0,    a[0],   d[1],    theta[1]],
                    [0.0,    a[1],   d[2],    theta[2]],
                    [-90.0,  a[2],   d[3],    theta[3]]   ])

print("DH Table:\n", DH, DH.shape)

# Create Math Objects
cos = tf.cos
sin = tf.sin
tan = tf.tan
arccos = tf.acos
arcsin = tf.asin
arctan = tf.atan    

# DH to Matrix
def dh2mat(dhparam):
    rotz = tf.Variable([    [cos(dhparam[3]), -sin(dhparam[3]), 0,  0],
                            [sin(dhparam[3]), cos(dhparam[3]),  0,  0],
                            [0,               0,                1,  0],
                            [0,               0,                0,  1]  ])
    
    transz = tf.Variable([  [1, 0,  0,  0],
                            [0, 1,  0,  0],
                            [0, 0,  1,  dhparam[2]],
                            [0, 0,  0,  1]  ])

    transx = tf.Variable([  [1, 0,  0,  dhparam[1]],
                            [0, 1,  0,  0],
                            [0, 0,  1,  0],
                            [0, 0,  0,  1]  ])
    
    rotx = tf.Variable([    [1,     0,                  0,                  0],
                            [0,     cos(dhparam[0]),    -sin(dhparam[0]),   0],
                            [0,     sin(dhparam[0]),    cos(dhparam[0]),    0],
                            [0,     0,                  0,                  1]  ])

    T = tf.linalg.matmul(rotx, transz)
    T = tf.linalg.matmul(T, transx)
    T = tf.linalg.matmul(T, rotz)
    return T

# Create the T matrices

T0 = tf.Variable([  [cos(theta[0]), -sin(theta[0]), 0,  0],
                    [sin(theta[0]), cos(theta[0]),  0,  0],
                    [0,             0,              1,  0],
                    [0,             0,              0,  1]  ])

print("T0:\n", T0)

T0prop = dh2mat(DH[0])

print("T0prop:\n", T0prop)

T01 = tf.Variable([ [cos(theta[1]), -sin(theta[1]), 0,  a[0]],
                    [sin(theta[1]), cos(theta[1]),  0,  0   ]  ,
                    [0,             0,              1,  0   ],
                    [0,             0,              0,  1   ]  ])

print("T1:\n", T01)

T01prop = dh2mat(DH[1])

print("T01prop:\n", T01prop)

T12 = tf.Variable([ [cos(theta[1]), -sin(theta[1]), 0,  a[1]],
                    [sin(theta[1]), cos(theta[1]),  0,  0   ]  ,
                    [0,             0,              1,  0   ],
                    [0,             0,              0,  1   ]  ])

print("T12:\n", T12)

T12prop = dh2mat(DH[2])

print("T12prop:\n", T12prop)

print("once")

while 1:
    pass    
