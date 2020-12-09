import serial 
import time
import os

def upload():
    # Begin Arduino programming via sketch.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Base Path: " + base_dir)
    sketch_name = "setupsketch"
    sketch_path = base_dir + "\\" + sketch_name
    port = "--port COM5" 
    command = "--upload"
    flags = "-v"

    command_string = "arduino" + " "                \
                    + command + " "                  \
                        + port + " "                \
                            + sketch_path + "\\"    \
                                + flags

    print("Command String: " + command_string)
    os.system(command_string)

class dueserial():
    
    def __init__(self):
        self = serial.Serial('COM5', 9600)

    def initialize_serial(self):
        # Serial communication objects.
        # dueserial = serial.Serial('COM5', 9600)

        # Initializers
        time.sleep(2)
        print (self.readline())

        print("Serial Established")

    def write_serial(self, data_out):
            self.write(data_out)