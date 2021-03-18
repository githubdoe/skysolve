#pi camera to be used with web based GUI plate solver
import io
import time
import picamera
from base_camera import BaseCamera
from fractions import Fraction

class Camera(BaseCamera):
    shutter = 10000000
    iso = 800
    frameSize = (2000,1000)
    theCam = None
    def __init__(self,frameSize, shutter,iso):
        BaseCamera.__init__(self)
        self.shutter = shutter
        self.iso = iso
        self.frameSize = frameSize
        
    @staticmethod
    def frames():
        with picamera.PiCamera() as camera:
            Camera.theCam = camera
            camera.resolution = Camera.frameSize
            camera.iso = 100
            camera.exposure_mode = 'auto'
            camera.framerate = Fraction(1,6)

            camera.shutter_speed = 1000000
            camera.exposure_mode = 'off'

            time.sleep(10)

            stream = io.BytesIO()

            for _ in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=False):
                # return current frame
                stream.seek(0)
                yield stream.read()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()



import os           
class FileCamera(BaseCamera):
    theCam = None
    file_names = []
    current = None
    ndx = 0
    print ('filecame')
    relevant_path = "/home/pi/pyPlateSolve/data"
    included_extensions = ['jpg','jpeg']
    file_names = [fn for fn in os.listdir(relevant_path)  if any(fn.endswith(ext) for ext in ['jpg'])]
    print ('files:', len(file_names))
    n = len(file_names)


        
    @staticmethod
    def frames():
        print('frames called')
        while True:
            fn = os.path.join(FileCamera.relevant_path, FileCamera.file_names[FileCamera.ndx % FileCamera.n])
            FileCamera.current = fn
            print ('yieding', fn)
            with open(fn, 'rb') as infile:
                yield infile.read()
            FileCamera.ndx += 1
            time.sleep(5)

