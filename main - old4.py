#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference


import queue
from numpy.core.defchararray import array
from numpy.core.numeric import full
import time
from pynput import mouse
import numpy as np 
from queue import Queue
from serial.serialutil import to_bytes


# Thread synchronization mechanism objects.
x_queue = Queue(256)
y_queue = Queue(256)
btn_queue = Queue(256)
prs_queue = Queue(256)
dx_queue = Queue(256)
dy_queue = Queue(256)

# Serial communication objects.
dueserial = serial.Serial('COM5', 9600)

# Mouse objects.
listener = mouse.Listener()

def on_move(x, y):
    try:
        x_queue.put_nowait(x)
        y_queue.put_nowait(y)
    except queue.Empty:
        pass
    except queue.Full:
        pass

def on_click(x, y, button, pressed):
    if not pressed:
        # Stop listener
        return  # False
    try:
        btn_queue.put_nowait(button)
        prs_queue.put_nowait(pressed)
    except queue.Empty:
        pass
    except queue.Full:
        pass

def on_scroll(x, y, dx, dy):
    try:
        dy_queue.put_nowait(dy)
    except queue.Empty:
        pass
    except queue.Full:
        pass

# Non-blocking mouse listener thread start.
# listener = mouse.Listener(
#     on_move=on_move,
#     on_click=on_click,
#     on_scroll=on_scroll)
# listener.start()

# Initializers
time.sleep(2)
print (dueserial.readline())
print("Serial Established")


# Lengths
# len[0][1] is the length from p0 p1
# len[1][2] is the length from p1 p2
# len[2][3] is the length from p2 p3
# len[0][2] is the length from p0 p2
# len[0][3] is the length from p0 p3
# len[1][3] is the length from p1 p3
len = [12.5, 12.5, 18.5]

# Mouse data variables
data = [0, 0]
data_last = [0, 0]

# Serial write packet.
data_out = [10, 90, 45, 180 ,180, 90, 10]

# Geometric variables
targets = [-13.08, 8.84, 90]

ptheta = [45, 180, 180]
reltheta = [45, 135, 225]
coordinates = [[0.0, 0.0], 
    [8.84, 8.84], 
    [0.0, 17.68], 
    [-13.08, 8.84]]

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

constraint = [[shoulder_min, shoulder_max], [elbow_min, elbow_max], [wristv_min, wristv_max]]

while 1:
    #PHysical Constraints
    xmax = 44.0
    xmin = -44.0
    ymin = -7.0
    ymax = 44.0

    try:
        data[0] = x_queue.get_nowait()
    except (queue.Full, queue.Empty):
        pass
    try:
        data[1] = y_queue.get_nowait()
    except (queue.Full, queue.Empty):
        pass
    try:
        btn = btn_queue.get_nowait()
    except (queue.Full, queue.Empty):
        pass
    try:
        press = prs_queue.get_nowait()
    except (queue.Full, queue.Empty):
        pass
    try:
        scroll = dy_queue.get_nowait()
    except (queue.Full, queue.Empty):
        pass

# X Y Filter
    if(data[0] > data_last[0]):
        if(targets[0] < xmax):
            targets[0] = targets[0] + 1
        else:
            targets[0] = xmax

    elif(data[0] < data_last[0]):
        if(targets[0] > xmin):
            targets[0] = targets[0] -1
        else:
            targets[0] = xmin

    if(data[1] > data_last[1]):
        if(targets[1] < ymax):
            targets[1] = targets[1] +1
        else:
            targets[1] = ymax

    elif(data[1] < data_last[1]):
        if(targets[1] > ymin):
            targets[1] = targets[1] - 1  
        else:
            targets[1] = ymin
    
    print("xtarg %d, ytarg %d" %(targets[0], targets[1]))

    targets[0] = int(input("xtarg:"))
    targets[1] = int(input("ytarg:"))


    def gettanangle(y2, x2, y1, x1):
        if((x2-x1) == 0):
            return (np.arctan(0) *(np.pi/180))
        else:
            return (np.arctan((y2-y1)/(x2-x1))*(np.pi/180))

    def angledelta(ytarg, xtarg, ox, oy, p3x, p3y):
        return gettanangle(ytarg, xtarg, oy, ox) - gettanangle(p3y, p3x, oy, ox)
    
    def getrelangle(i):
        if(i == 0):
            return ptheta[0]
        else:
         return (reltheta[i-1] - 90) + ptheta[i]

    def update_angles(i, ytarg, xtarg):
        if( i == 3):
            return
        ptheta[i] = ptheta[i] + angledelta(ytarg, xtarg, coordinates[i][0], coordinates[i][1], coordinates[3][0], coordinates[3][1])
        print("Angle Update:", i, ytarg, xtarg,coordinates[i][0], coordinates[i][1], coordinates[3][0], coordinates[3][1], ptheta[i])
        if(ptheta[i] > constraint[i][1]):
            ptheta[i] = constraint[i][1]
        elif(ptheta[i] < constraint[i][0]):
            ptheta[i] = constraint[i][0]
        for i in range(i, 2, 1):
            reltheta[i] = getrelangle(i)        
        return


    def update_coordinates(i):
        # Update whichever point is at focus.
        for i in range(i, 3, 1):
            if(i == 0):
                break
            else:
                coordinates[i][0] = coordinates[i-1][0] + (np.cos(reltheta[i-1])*len[i-1])
                coordinates[i][1] = coordinates[i-1][1] + (np.cos(reltheta[i-1])*len[i-1])
                print("CoordinateUPdate:", i, coordinates[i][0], coordinates[i][1])

    for i in range(0, 3, 1):
        update_angles(i, targets[1], targets[0])
        update_coordinates(i)

    data_out[2] = int(ptheta[0])
    data_out[3] = int(ptheta[1])
    data_out[4] = int(ptheta[2])

    # i = int(input("Servo Select"))

    # data_out[i] = int(input("Servo Input"))

    print(" Data_out: %d,%d,%d,%d,%d,%d,%d" %(data_out[0], data_out[1],data_out[2],data_out[3],data_out[4],data_out[5],data_out[6]))

    dueserial.write(data_out)

    data_last = data.copy()