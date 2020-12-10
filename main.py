#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

from numpy.core.defchararray import array
from numpy.core.numeric import full
import numpy as np 
import coms
from math import pi
from math import radians as rad
from math import degrees as deg
import esr

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
a = np.array([12.5, 12.5, 18.5])

# X axis offset
d = np.array([0.0, 0.0, 0.0, 0.0])

# Joint values
theta = np.array([rad(90.0), rad(90.0), rad(90.0), rad(90.0)])

# Axis Twist angles
alpha = np.array([rad(0.0), rad(0.0), rad(0.0), rad(-90.0)])

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

constraint = np.array([  [shoulder_min, shoulder_max],             
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
# [joint] =         [alphai, ai,   di,      thetai]
DH = np.array([     [alpha[0],    0.0,    d[0],    theta[0]],
                    [alpha[1],    a[0],   d[1],    theta[1]],
                    [alpha[2],    a[1],   d[2],    theta[2]],
                    [alpha[3],    a[2],   d[3],    theta[3]]   ])

print("\n DH Table:\n", DH, DH.shape)

# Create Trig objects (operate in Rads)
cos = np.cos
sin = np.sin
tan = np.tan
arccos = np.arccos
arcsin = np.arcsin
arctan = np.arctan

# World and tool coordinates
s0 = np.array( [[0.0],[0.0],[0.0], [1.0]] )
s1 = np.array( [[0.0],[0.0],[0.0], [1.0]] )
s2 = np.array( [[0.0],[0.0],[0.0], [1.0]] )
s3 = np.array( [[0.0],[0.0],[0.0], [1.0]] )

s = np.array([s0, s1, s2, s3]) 

# DH to Matrix.
# Result is the DH Matrix of the previous joint:An, also known as nTn+1
def An(n):

    # Rotation around the Z axis using thetai
    rotz = np.array([    [cos(theta[n]), -sin(theta[n]), 0,  0],
                            [sin(theta[n]), cos(theta[n]),  0,  0],
                            [0,               0,                1,  0],
                            [0,               0,                0,  1]  ])
    
    # Translation along the Z axis using ai
    transz = np.array([  [1, 0,  0,      0],
                            [0, 1,  0,      0],
                            [0, 0,  1,      a[n]],
                            [0, 0,  0,      1]  ])

    # Translation along the X axis using di.
    transx = np.array([  [1, 0,  0,      d[n]],
                            [0, 1,  0,      0],
                            [0, 0,  1,      0],
                            [0, 0,  0,      1]  ])
    
    # Rotation about the X axis using alphai.
    rotx = np.array([    [1,     0,                  0,                  0],
                            [0,     cos(alpha[n]),    -sin(alpha[n]),   0],
                            [0,     sin(alpha[n]),    cos(alpha[n]),    0],
                            [0,     0,                  0,                  1]  ])

    T = np.matmul(rotz, transz)
    T = np.matmul(T, transx)
    T = np.matmul(T, rotx)

    print("\n Return T%d:\n" % n, T)

    return T

s[1] = np.matmul(An(0), s[0])
s[2] = np.matmul(An(1), s[1])
s[3] = np.matmul(An(2), s[2])


print("\n s3:\n", s)

print("once")

while 1:
    break    
