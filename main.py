#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference


import queue
from numpy.core.defchararray import array
from numpy.core.numeric import full
import time
from pynput import mouse
import numpy as np 
from queue import Queue
from tensorflow._api.v2 import math

from tensorflow.python.ops.gen_math_ops import asin
import coms
import tensorflow as tf
from math import pi
from math import radians as rad
from math import degrees as deg

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
    print("\n Unable to connect to serial device.")

# Lengths
# 1st arm from shoulder, forecep, wrist with gripper length.
# a = tf.Variable([0.0, 12.5, 12.5, 18.5])
a = tf.constant([0.0, 12.5, 12.5, 18.5])

# X axis offset
# d = tf.Variable([0.0, 0.0, 0.0, 0.0])
d = tf.constant([0.0, 0.0, 0.0, 0.0])

# Joint values
# theta = tf.Variable([rad(45.0), rad(180.0), rad(180.0), rad(90.0)])
theta = tf.Variable([rad(45.0), rad(180.0), rad(180.0), rad(90.0)])

# Axis Twist angles
# alpha = tf.constant([rad(0.0), rad(0.0), rad(0.0), rad(-90.0)])
alpha = tf.constant([rad(0.0), rad(0.0), rad(0.0), rad(-90.0)])

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

print("\n Constraints:\n", constraint)

# DH Matrix 
# thetai: A rotation angle between two links, about the z-axis
# di: The distance(offset) on the z-axis, between links(prismatic)
# ai: The length of each common normal (Joint offset)
# alphai: The twist angle between two successive z-axes (Joint twist/revolute)
# Individual rows are of type "tf.Tensor"
# [joint] = 
#                   [alphai, ai,   di,      thetai]
# DH = tf.Variable([  [0.0,    0.0,    d[0],    theta[0]],
#                     [0.0,    a[0],   d[1],    theta[1]],
#                     [0.0,    a[1],   d[2],    theta[2]],
#                     [-(pi/2),  a[2],   d[3],    theta[3]]   ])

#                   [alphai, ai,   di,      thetai]
DH = tf.Variable([  [alpha[0],    0.0,    d[0],    theta[0]],
                    [alpha[1],    a[0],   d[1],    theta[1]],
                    [alpha[2],    a[1],   d[2],    theta[2]],
                    [alpha[3],    a[2],   d[3],    theta[3]]   ])

print("\n DH Table:\n", DH, DH.shape)

# Create Trig objects (operate in Rads)
cos = tf.cos
sin = tf.sin
tan = tf.tan
arccos = tf.acos
arcsin = tf.asin
arctan = tf.atan


# DH to Matrix.
# Result is the DH Matrix of the previous joint:An, also known as nTn+1
def An(AnPlus1):

    # Rotation around the Z axis using thetai
    rotz = tf.Variable([    [cos(AnPlus1[3]), -sin(AnPlus1[3]), 0,  0],
                            [sin(AnPlus1[3]), cos(AnPlus1[3]),  0,  0],
                            [0,               0,                1,  0],
                            [0,               0,                0,  1]  ])
    
    # Translation along the Z axis using ai
    transz = tf.Variable([  [1, 0,  0,      0],
                            [0, 1,  0,      0],
                            [0, 0,  1,      AnPlus1[2]],
                            [0, 0,  0,      1]  ])

    # Translation along the X axis using di.
    transx = tf.Variable([  [1, 0,  0,      AnPlus1[1]],
                            [0, 1,  0,      0],
                            [0, 0,  1,      0],
                            [0, 0,  0,      1]  ])
    
    # Rotation about the X axis using alphai.
    rotx = tf.Variable([    [1,     0,                  0,                  0],
                            [0,     cos(AnPlus1[0]),    -sin(AnPlus1[0]),   0],
                            [0,     sin(AnPlus1[0]),    cos(AnPlus1[0]),    0],
                            [0,     0,                  0,                  1]  ])

    T = tf.linalg.matmul(rotz, transz)
    T = tf.linalg.matmul(T, transx)
    T = tf.linalg.matmul(T, rotx)

    return T

# Create the T matrices

T0 = tf.Variable([  [cos(theta[0]), -sin(theta[0]), 0,  0],
                    [sin(theta[0]), cos(theta[0]),  0,  0],
                    [0,             0,              1,  0],
                    [0,             0,              0,  1]  ])

print("\n T0:\n", T0)

T0prop = An(DH[0])

print("\n T0prop:\n", T0prop)

T01 = tf.Variable([ [cos(theta[1]), -sin(theta[1]), 0,  a[0]],
                    [sin(theta[1]), cos(theta[1]),  0,  0   ],
                    [0,             0,              1,  0   ],
                    [0,             0,              0,  1   ]  ])

print("\n T1:\n", T01)

T01prop = An(DH[1])

print("\n T01prop:\n", T01prop)

T12 = tf.Variable([ [cos(theta[1]), -sin(theta[1]), 0,  a[1]],
                    [sin(theta[1]), cos(theta[1]),  0,  0   ],
                    [0,             0,              1,  0   ],
                    [0,             0,              0,  1   ]  ])

print("\n T12:\n", T12)

T12prop = An(DH[2])

print("\n T12prop:\n", T12prop)

RTH = tf.linalg.matmul(T12, T01)
RTH = tf.linalg.matmul(T0, RTH)



print("\n RTH:\t", RTH)

print("once")

while 1:
    break    
