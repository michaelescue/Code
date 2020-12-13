#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

import numpy as np 
from math import pi
from math import radians as rad
import sympy as sy

# Lengths
# 1st arm from shoulder, forecep, wrist with gripper length.
a = np.array([12.5, 12.5, 18.5])

# X axis offset
d = np.array([0.0, 0.0, 0.0])

# Axis Twist angles
alpha = np.array([rad(0.0), rad(0.0), rad(0.0)])

# Create Trig objects (operate in Rads)
cos = sy.symbol()
sin = np.sin
tan = np.tan
arccos = np.arccos
arcsin = np.arcsin
arctan = np.arctan


# Initial frame coordinates
s0 = np.array( [[0.0],[0.0],[0.0], [1.0]] )

# Forward transform for joint 3
def s3(theta0, theta1, theta2):
    
    # Servo Orientation Corrections
    theta0 = theta0
    theta1 = theta1 - 90
    theta2 = theta2 - 90
    
    # Joint values
    theta = np.array([rad(theta0), rad(theta1), rad(theta2)])

    return  np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(rotz(0,theta),transx(0,theta)), rotz(1,theta)), transx(1,theta)), rotz(2,theta)), transx(2,theta))

# Fowardr transform for joint 2
def s2(theta0, theta1, theta2):
    
    # Servo Orientation Corrections
    theta0 = theta0
    theta1 = theta1 - 90
    theta2 = theta2 - 90
    
    # Joint values
    theta = np.array([rad(theta0), rad(theta1), rad(theta2)])

    return  np.matmul(np.matmul(np.matmul(rotz(0,theta),transx(0,theta)), rotz(1,theta)), transx(1,theta))

# Forward transform for joint 1
def s1(theta0, theta1, theta2):
    
    # Servo Orientation Corrections
    theta0 = theta0
    theta1 = theta1 - 90
    theta2 = theta2 - 90
    
    # Joint values
    theta = np.array([rad(theta0), rad(theta1), rad(theta2)])

    return  np.matmul(rotz(0,theta),transx(0,theta))

# Rotation around the Z axis using thetai
def rotz(n, theta):
    rotz = np.array([       [cos(theta[n]), -sin(theta[n]), 0,  0],
                            [sin(theta[n]), cos(theta[n]),  0,  0],
                            [0,               0,                1,  0],
                            [0,               0,                0,  1]  ])
    return rotz

# Translation on the x axis using ai
def transx(n, theta):
    transx = np.array([     [1, 0,  0,      a[n]],
                            [0, 1,  0,      0],
                            [0, 0,  1,      0],
                            [0, 0,  0,      1]  ])    
    return transx

