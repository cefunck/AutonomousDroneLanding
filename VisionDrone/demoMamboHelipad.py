"""
Demo of the ffmpeg based mambo vision code (basically flies around and saves out photos as it flies)

Author: Amy McGovern
"""
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
from SymbolTracker import *
import threading
import cv2
import time


# set this to true if you want to fly for the demo
testFlying = False


class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.latest_frame = None

    def store_frame(self, args):
        print("in save pictures on image %d " % self.index)

        new_frame = self.vision.get_latest_valid_picture()

        if new_frame is not None:
            self.latest_frame = new_frame



# you will need to change this to the address of YOUR mambo
mamboAddr = "e0:14:d0:63:3d:d0"

# make my mambo object
# remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)

    print("Preparing to open vision")
    mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=5)
    userVision = UserVision(mamboVision)
    mamboVision.set_user_callback_function(userVision.store_frame, user_callback_args=None)
    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)

    if (success):
        while True:
            frame = userVision.latest_frame
            if frame is not None:
                # Our operations on the frame come here
                painted_frame = SymbolTracker.track_and_paint_helipad(frame)

                # Display the resulting frame
                cv2.imshow('painted_frame', painted_frame)
                cv2.imshow('frame', frame)

            mambo.smart_sleep(1)

        # done doing vision demo
        print("Ending the sleep and vision")
        mamboVision.close_video()

        mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()
