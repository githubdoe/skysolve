#pi camera to be used with web based GUI plate solver  For RPI5 and using the picamera2 library
"""This is the camera interface to the PI camera_config
Picamera2 is the library that talks to the camera.  
This module setups up and uses that library.
It setups so that the camera is running taking images at the requested specs.
When a client requests a frame it gets it from the camera that has already
buffered the last fame.  So all that is needed is the request the current
frame from the camera and signal the client that it is ready.
"""
from picamera2 import Picamera2, Preview
from libcamera import ColorSpace
import time
import io
import time
import pprint
from tkinter import EXCEPTION
 

from fractions import Fraction
import threading
from _thread import get_ident

from picamera2 import Picamera2


class imageEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self, tt = 60.):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait(timeout = tt)

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()

"""  my new py camera class
    does not derive from a base class
    needs init to start frame thread
    needs __del__ method to kill that thread  maybe in a close function

    Don't think any static methods are needed.

    thread camera gain init function and call in init 


"""

class skyCamera():
    camera = None
    event = imageEvent()
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    shutter = 1000000
    ISO=800
    resolution= (2000,1500)
    abortMode = False
    runMode = True
    cameraStopped = True
    format = 'jpeg'
    def __init__(self, skystatus, shutter=.9, ISO=800, resolution=(2000,1500), format = 'jpeg'):
        print("skystatus", skystatus, shutter)

        self.skyStatus = skystatus

        self.camera = Picamera2() 

        capture_config = self.camera.create_still_configuration(main={'size':resolution, 'format': 'RGB888'})
        #self.config = capture_config = self.camera.create_still_configuration(colour_space=ColorSpace.Sycc(),main={'size':resolution})
        self.camera.configure(capture_config)
        #pprint.pprint(self.camera.sensor_modes)
        #self.camera.set_controls({"ExposureTime": 100, "AnalogueGain": ISO/100, "AwbEnable": True})
        self.camera.controls.ExposureTime = int(1000000 * shutter)
        self.camera.controls.AnalogueGain = ISO/100
        self.camera.exposure_mode = 'off'
        self.camera.controls.AeEnable = False
        try:
            pass
            #self.camera.controls.AwbEnable = False
        except:
            pass
        self.camera.start()
        time.sleep(1)
        #self.camera.controls.ExposureTime = int(100000 * shutter)



        #print('config',self.config)
        
        # self.camera.preview_configuration.main.size = resolution
        # self.camera.preview_configuration.main.align()

        #self.camera.awb_gains = (1.8,1.8)




        #self.camera.controls.ExposureTime = int(100000 * shutter)
        #self.camera.controls.AnalogueGain = ISO/100.
        #print("shutter", self.camera.controls.ExposureTime, shutter)
        # self.camera.configure(self.config)

        #print("\n\nproperities",self.camera.camera_properties)


        #print('config after start', self.camera.stream_configuration("main"))


        #self.camera.resolution = resolution
        #self.camera.framerate = Fraction(1,6)
        print('controlssssss',self.camera.camera_controls)
        self.ISO=ISO
        self.resolution=resolution
        self.format = format
        #self.setupGain()
        self.count = 0
        #self.thread = threading.Thread(target=self._thread)
        #self.thread.start()



    def __del__(self):
        if self.camera:
            self.camera.close()
            
    def pause(self):
        self.runMode = False

    def resume(self):
        self.runMode = True

    def setISO(self, iso):
        self.ISO = iso
        self.setupGain(iso/100.)

    def status(self):
        return [self.ISO, self.camera.shutter_speed, self.resolution]

    def setResolution(self,  size):
        self.camera.stop()
        res = [int(x) for x in size.split('x')]
        capture_config = self.camera.create_still_configuration(main={'size':res, 'format': 'RGB888'})
        #self.config = capture_config = self.camera.create_still_configuration(colour_space=ColorSpace.Sycc(),main={'size':resolution})
        self.camera.configure(capture_config)

        self.camera.start()
        time.sleep(1)


    def setFormat(self,type):
        self.format = type
        self.runMode = False
        while not self.cameraStopped:
            time.sleep(.2)
        self.runMode = True

    def setShutter(self, value):
        self.shutter = value
        self.camera.controls.ExposureTime = int(1000000 * value)
        print ("new shutter speed", value, flush=True)

    def setupGain(self, value):
        with self.camera.controls as controls:
            controls.AnalogueGain = value


    

    def get_frame(self):
        """Return the current camera frame."""
        return self.camera.capture_array("main")


   
if __name__ == '__main__':

    def delayedStatus(delay, status):
        print("status",status)
    from pprint import *
    import cv2
    class dcam():
        def __init__(self,cam):
            self.camera = cam
    """   cam = dcam( Picamera2())
    ISO = 800

    capture_config = cam.camera.create_still_configuration(main={'size':(600,600), 'format': 'RGB888'})
    cam.camera.configure(capture_config)
    cam.camera.set_controls({ "AnalogueGain": ISO/100, "AwbEnable": True})

    cam.camera.controls.AnalogueGain = 1.0
    cam.camera.start()
    time.sleep(1) """


    #quit()
    #shutter = 100000
    #cam.camera.controls.ExposureTime = shutter
    #print("xxxx",cam.camera.controls.ExposureTime)
    cam = skyCamera( delayedStatus, shutter=.1, ISO=800, resolution=(800,900), format = 'jpeg')
    #print ("size size",cam.camera.stream_configuration("main"))
    size = cam.camera.capture_metadata()['ScalerCrop'][2:]
    print('meta',cam.camera.capture_metadata())
    full_res = cam.camera.camera_properties['PixelArraySize']
    offset = [0,0]

    while True:
        im = cam.camera.capture_array('main')

        cv2.imshow("Camera", im)
        ch = cv2.waitKey(1)
        if ch != -1:
            print ("ch", ch)
            if ch == ord('q'):
                print("Quitting")
                break
            elif ch == ord('z'):
                size = [int(s * 0.95) for s in size]
                offset = [(r - s) // 2 for r, s in zip(full_res, size)]
                print ('zoomed',offset,size, offset+size)
                cam.camera.set_controls({"ScalerCrop": offset + size})

            elif ch == ord('x'):
                size = [int(s * (1./0.95)) for s in size]
                offset = [(r - s) // 2 for r, s in zip(full_res, size)]
                cam.camera.set_controls({"ScalerCrop": offset + size})
                print ('zoomednnn',offset,size, offset+size)
            elif ch == 81:   #<--
                size = [int(s) for s in size]
                offset[0] = offset[0] + int(size[0]/25)
                cam.camera.set_controls({"ScalerCrop": offset + size})
                print("offset",offset)
            elif ch == 83:   # ->
                    
                offset[0] = offset[0] - int(size[0]/25)
                if offset[0] < 0:
                    offset[0] = 0
                cam.camera.set_controls({"ScalerCrop": offset + size})
                print("offset",offset)
            elif ch == ord('a'):
                cam.camera.set_controls({"ScalerCrop": offset + size})
            elif ch == ord('b'): #Brigher
                shutter = int(shutter * 1.25)
                print('shutter',shutter/100000)
                cam.camera.controls.ExposureTime = shutter

            elif ch == ord('d'): #Brigher
                shutter = int(shutter / 1.25)
                cam.camera.controls.ExposureTime = shutter

            else:

                print("shape", im.shape)
                print ("cam controls",cam.camera.camera_controls)
                print("main config raw",cam.camera.camera_configuration()['raw'])
                print("main config main",cam.camera.camera_configuration()['main'])
                cam.camera.controls.ExposureTime = 10000
                print("xxxx",cam.camera.controls.ExposureTime)


    #print("properities",picam.camera_properties)
    #pprint(picam.camera_controls)

