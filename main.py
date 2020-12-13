#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference

import numpy as np 
import coms
from forward import s3


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

bigarray = np.array([[0, 0, 0, 0, 0]])
print(bigarray)
xrange = 43.5
yrange = 43.5

# Float values 0 to 180
theta = np.arange(start=0, stop=181, step=1, dtype=int)

print(theta)
 
# for x in range(0,1):
#     for y in range(0,1):
#         for z in range(0,1):
#             s = s3(theta[x], theta[y], theta[z])
#             bigarray = np.append(  bigarray, 
#                         [[s[0], s[1], theta[x], theta[y], theta[z]]],
#                         axis=0
#             )
#             print(s[0], s[1], theta[x], theta[y], theta[z])
#             print(bigarray)

for x in range(theta.shape[0]):
    print(x)
    for y in range(theta.shape[0]):
        for z in range(theta.shape[0]):
            s = s3(theta[x], theta[y], theta[z])
            bigarray = np.append(  bigarray, 
                        [[s[0], s[1], theta[x], theta[y], theta[z]]],
                        axis=0
            )
            # print(s[0], s[1], theta[x], theta[y], theta[z])
            # print(bigarray)