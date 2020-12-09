#https://circuitdigest.com/microcontroller-projects/arduino-python-tutorial
#https://pynput.readthedocs.io/en/latest/mouse.html#reference


import queue
from numpy.core.defchararray import array
from numpy.core.numeric import full
import serial
import time
from pynput import mouse
import numpy as np 
from queue import Queue



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
listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)
listener.start()

# Initializers
time.sleep(2)
print (dueserial.readline())
print("Serial Established")

#PHysical Constraints
xmax = 44
xmin = -44
ymin = -7
ymax = 44
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

# Lengths
# len[0][1] is the length from p0 p1
# len[1][2] is the length from p1 p2
# len[2][3] is the length from p2 p3
# len[0][2] is the length from p0 p2
# len[0][3] is the length from p0 p3
# len[1][3] is the length from p1 p3
len = [[0.0, 12.50, 17.68, 15.79], 
    [12.50, 0.0, 12.50, 21.92], 
    [17.68, 12.50, 0.0, 18.50], 
    [15.79, 21.92, 18.50, 0.0]]

# Mouse data variables
data = [0, 0]
data_last = [0, 0]

# Serial write packet.
data_out = bytearray([10, 90, 45, 180 ,180, 90, 10])

# Geometric variables
x = 0
y = 0
ptheta = [45, 180, 180]
ree = 18.5
coordinates = [[0.0, 0.0], 
    [8.84, 8.84], 
    [0.0, 17.68], 
    [-13.08, 8.84]]
theta_p0p3 = 0
theta_p0targ = 0

# Index
i = 0

while 1:

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
        if(x < xmax):
            x = x + 1
        else:
            x = xmax

    elif(data[0] < data_last[0]):
        if(x > xmin):
            x = x - 1
        else:
            x = xmin

    if(data[1] > data_last[1]):
        if(y < ymax):
            y = y + 1
        else:
            y

    elif(data[1] < data_last[1]):
        if(y > ymin):
            y = y - 1  
        else:
            y = ymin

    def gettanangle(ypn, xpn, yee, xee):
        return np.arctan((ypn-yee)/(xpn-xee))

    def sumptheta(i, ptheta):
        sumofangle = 0
        if(i != 0):
            for x in range (i-1, 0, -1):
                sumofangle = sumofangle + ptheta[x]
        return sumofangle

    #used to calculate difference in angles
    def p3angle(ypn, xpn, yee, xee, i):
       return gettanangle(ypn, xpn, yee, xee)+sumptheta(i)

    def targangle(ypn, xpn, ytar, xtar, i):
        return p3angle(ypn, xpn, ytar, xtar, i)

    def differenceangle(ypn, xpn, yee, xee, ytar, xtar, i):
        return (targangle(ypn, xpn, ytar, xtar, i) - p3angle(ypn, xpn, yee, xee, i))

    def update_coordinates(i):
        # Update whichever point is at focus.
        if(i != 0):

            segment_length = len[i]
            coordinates[i][0] = coordinates[i-1][0] + (np.cos(sumptheta(i, ptheta))*segment_length)
            coordinates[i][1] = coordinates[i-1][1] + (np.cos(sumptheta(i, ptheta))*segment_length)
        # Update EndEff Coordinates.
        coordinates[2][0] = 

    

    dueserial.write(data_out)

    data_last = data.copy()

