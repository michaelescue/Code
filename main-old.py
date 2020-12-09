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

#Constants
xmax = 44
xmin = -44
ymin = -7
ymax = 44

# Queues for thread sync.
x_queue = Queue(256)
y_queue = Queue(256)
btn_queue = Queue(256)
prs_queue = Queue(256)
dx_queue = Queue(256)
dy_queue = Queue(256)

# Serial object.
dueserial = serial.Serial('COM5', 9600)

# Sleep delay for serial sync.
time.sleep(2)

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
print (dueserial.readline())
print("Serial Established")

data = [0, 0]
data_last = [0, 0]
data_out = bytearray([10, 90, 45, 180 ,180, 90, 10])
i = 1

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

# Btn Logic
    # None

# Scroll Logic
    if(scroll == 1):
        if(i < 5):
            i = i + 1

    if(scroll == -1):
        if(i > 0):
             i = i - 1
        

# Rotation Limit Logic
    if(i == 1):
        min = 0
        max = 180
    if(i == 2):
        min = 15
        max = 165
    if(i == 3):
        min = 0
        max = 180
    if(i == 4):
        min = 0
        max = 180
    if(i == 5):
        min = 0
        max = 180
    if(i == 6):
        min = 10
        max = 73


# X Y Logic
    if((i == 1) or (i == 4) or (i == 5)):
        if(data[0] > data_last[0]):
            if(data_out[i] < max):
                data_out[i] = data_out[i] + 1
            else:
                data_out[i] = max

        elif(data[0] < data_last[0]):
            if(data_out[i] > min):
                data_out[i] = data_out[i] - 1
            else:
                data_out[i] = min

    if((i == 2) or (i == 3) or (i == 4)):
        if(data[1] > data_last[1]):
            if(data_out[i+2] < 180):
                data_out[i+2] = data_out[i+2] + 1
            else:
                data_out[i+2]

        elif(data[1] < data_last[1]):
            if(data_out[i+2] > 0):
                data_out[i+2] = data_out[i+2] - 1  
            else:
                data_out[i+2] = 0

# Gripper Logic
    if(press):
        data_out[6] = 73
    else:
        data_out[6] = 10

    dueserial.write(data_out)

    data_last = data.copy()

