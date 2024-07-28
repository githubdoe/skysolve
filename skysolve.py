from subprocess import call
from zipfile import ZipFile
from flask import Flask, render_template, request, Response, send_file, send_from_directory
from flask import stream_with_context


import time
from flask.wrappers import Request
import pprint
import copy
from camera_pi import skyCamera

from datetime import datetime, timedelta
import threading
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import math
import json
from shutil import copyfile
from enum import Enum, auto
import io
import getpass
import copy
import sys
sys.path.append('/home/pi/cedar/cedar-solve/tetra3')
print('PYTHONPath is',sys.path)
#from tetra3old import Tetra3
from tetra3 import Tetra3, get_centroids_from_image
#import cedar_detect_client
#import cedar_detect_pb2, cedar_detect_pb2_grpc
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
import cv2
import glob
import logging
from collections import deque
from time import perf_counter as precision_timestamp
debugLastState = 'startup'

solveReadyForImage = False
solveImage = None
GUIReadyForImage = False
GUIImage = None

print("argssss", sys.argv, len(sys.argv))
print('user', getpass.getuser())
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.IN, pull_up_down=GPIO.PUD_UP)
initGPIO7 = GPIO.input(7)
if ( not initGPIO7):
    try:
        os.system('sudo accesspopup -a')
    except BaseException as e:
        print('could not switch to access point')
    
print("gpio", initGPIO7)
try:
    os.system('systemctl restart encodertoSkySafari.service ')
except BaseException as e:
    print("did not start encoder", e )

usedIndexes = {}

# Create instance and load default_database (built with max_fov=12 and the rest as default)
t3 = None
print("what is t3", t3)
if t3 == None:
    t3 = Tetra3('default_database')
    #cedar_detect = cedar_detect_client.CedarDetectClient()
    print("tetra t3", t3)
print('after tetra')
class Mode(Enum):
    PAUSED = auto()
    ALIGN = auto()
    SOLVING = auto()
    PLAYBACK = auto()
    SOLVETHIS = auto()

    # playback and sovlve history images without user clicking on each.
    AUTOPLAYBACK = auto()





app = 30
solving = False
maxTime = 50
searchEnable = False
searchRadius = 90
solveLog = deque(maxlen=1000)
ra = 0
dec = 0
solveStatus = ''
computedPPa = ''



focusStd = ''
state = Mode.ALIGN

testNdx = 0
lastDisplayedFile = ''
testFiles = []
root_path = os.getcwd()
if len(sys.argv) == 2:
    root_path = sys.argv[1]


solve_path = os.path.join(root_path, 'static')
capPath = os.path.join(solve_path, 'cap.jpeg')
guiPath = os.path.join(solve_path, 'gui.jpeg')
history_path = os.path.join(solve_path, 'history')
doDebug = True
logging.basicConfig(filename=os.path.join(solve_path,'skysolve.log'),\
     format='%(levelname)s %(asctime)s %(message)s',filemode='w',level=logging.WARNING)
if not os.path.exists(history_path):
    os.mkdir(history_path)
demo_path = os.path.join(solve_path, 'demo')
test_path = '/home/pi/pyPlateSolve/data'
solveThisImage = ''

solveCurrent = False
triggerSolutionDisplay = False
saveObs = False

obsList = []
ndx = 0
lastObs = ""

cameraNotPresent = True

"""
skyConfig = {'camera': {'shutter': 1, 'ISO':800, 'frame': '800x600','format': 'jpeg'},
    'solver':{'currentProfile': 'default', 'startupSolving': False},
    'observing':{'saveImages': False, 'showSolution': False, 'savePosition': True , 'obsDelta': .1},
    'solverProfiles' :{\
  
        'default': {'name': 'default', 'maxTime': 20, 'solveSigma':5, 'solveDepth':20, 'UselastDelta':False, 'FieldWidthMode':'FieldWidthModeField',
          'FieldWidthModeaPP': 'checked', 'FieldWidthModeField':"", 'FieldWidthModeOther':'', 'aPPLoValue': 27, 'aPPHiValue': 30,
          'fieldLoValue': 14, 'fieldHiValue': 0, 'searchRadius': 10,  'solveVerbose':False , "showStars": True ,"plots":True},
        
        '25FL' : {'name': '25FL', 'maxTime': 20, 'solveSigma':9, 'solveDepth':20, 'UselastDelta':False, 'FieldWidthMode':'FieldWidthModeField',
           'FieldWidthModeaPP': 'checked', 'FieldWidthModeField':"", 'FieldWidthModeOther':'', 'aPPLoValue': 27, 'aPPHiValue': 0,
            'fieldLoValue': 14, 'fieldHiValue': 0, 'searchRadius': 10,  'solveVerbose':False , "showStars": False ,"plots":False},
         
        '50FL': {'name':'50FL','maxTime': 20, 'solveSigma':9, 'solveDepth':20, 'UselastDelta':False, 'FieldWidthMode':'FieldWidthModeField',
           'FieldWidthModeaPP': 'checked', 'FieldWidthModeField':"", 'FieldWidthModeOther':'', 'aPPLoValue': 40, 'aPPHiValue': 0,
            'fieldLoValue': 22, 'fieldHiValue': 0, 'searchRadius': 10,  'solveVerbose':False , "showStars": False ,"plots":False}}}
"""
import builtins
class ListStream:
    def write(self, s):
        global doDebug
        if doDebug:
            logging.warning(s)
    def flush(self):
        pass




def saveConfig():
    with open('skyConfig.json', 'w') as f:

        json.dump(skyConfig, f, indent=4)
format

# saveConfig()
print('cwd', os.getcwd())


with open(os.path.join(root_path, 'skyConfig.json')) as f:
    skyConfig = json.load(f)
print(json.dumps(skyConfig['solverProfiles']['25FL'], indent=4))
print(json.dumps(skyConfig['observing'], indent=4))
print(json.dumps(skyConfig['camera'], indent=4) )

imageName = 'cap.'+skyConfig['camera']['format']
capPath = os.path.join(solve_path, imageName)
#print (skyConfig)
skyCam = None
skyStatusText = ''
verboseSolveText = ''


def delayedStatus(delay, status):
    global skyStatusText
    time.sleep(delay)
    skyStatusText = status


def setupCamera():
    global skyCam, cameraNotPresent, state, skyStatusText
    if not skyCam:

        try:
            skyCam = skyCamera(delayedStatus, shutter=
                float(skyConfig['camera']['shutter']),
                format='RGB888',
                resolution= [int(x) for x in skyConfig['camera']['frame'].split('x')])
            cameraNotPresent = False
            if skyConfig['solver']['startupSolveing']:
                print("startup in solving")
                state = Mode.SOLVING
            startFrame = skyCam.get_frame()
            if startFrame is None:
                print("camera did not seem to start")
            print("camera started and frame received",
                  cameraNotPresent )
        except Exception as e:
            print(e)
            cameraNotPresent = True
            skyStatusText = 'camera not connected or enabled.  Demo mode and replay mode will still work however.'

def getImage():
    global  skyCam
    frame = skyCam.camera.capture_array('main')
    return frame
    
# obsolete can be removed later
def frameGrabberThread():
    global solveReadyForImage, GUIReadyForImage, solveImage, GUIImage,skyCam
    while(True):
        frame = skyCam.camera.capture_array('main')
        if solveReadyForImage:
            #print('getting solve image')
            solveImage = copy.deepcopy(frame)
            #print('image shape', solveImage.shape)

            #convert to black and white
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print('gray image', gray_image.shape)
            solveImage = copy.deepcopy(gray_image)
            solveReadyForImage = False
        if GUIReadyForImage:
            #print('getting gui image')
            GUIImage = JPEG = cv2.imencode('.jpeg', frame)[1].tobytes()
            GUIReadyForImage = False
setupCamera()
#frameGrabberT = threading.Thread(target=frameGrabberThread)
#frameGrabberT.start()
framecnt = 0


lastsolveTime = datetime.now()
justStarted = True
camera_Died = False
solveCompleted = False

lastpictureTime = -1
framecnt = 0


def delayedStatus(delay,status):
    global skyStatusText
    time.sleep(delay)
    skyStatusText = status





def shutThread():
    time.sleep(3)
    call("sudo nohup shutdown -h now", shell=True)

#thread to watch for the switch to change positions for 10 seconds then shutdown
def switchWatcher():
    
    #check for shutdown switch throw

    initGPIO7 = GPIO.input(7)
    while True:
        time.sleep(2)
        pin7 = GPIO.input(7)
        if pin7 != initGPIO7:
            print("pin7 changed states")
            time.sleep(3)
            if GPIO.input(7) == pin7:
                print("shutting down")
                skyStatusText = "shutting down."
                th = threading.Thread(target=shutThread)
                state = Mode.PAUSED
                th.start()
                break

switcherWatcherThread  = threading.Thread(target=switchWatcher)
switcherWatcherThread.start()
#
#  this is responsible for getting images from the camera even in align mod
def solveThread():
    global skyStatusText, focusStd, solveCurrent, state, skyCam,  testNdx, camera_Died,\
        solveLog, solveCompleted, debugLastState, lastpictureTime,\
            solveReadyForImage,solveImage,solveThisImage

    # save the image to be solved in the file system  for the gen() routine to give to the client browser
    def saveImage(frame):
        global doDebug
        #print("saving image using frame", solve_path)
        try:
            cv2.imwrite(os.path.join(solve_path, 'cap.jpg'),frame)
            return True

        except Exception as e:

            if doDebug:
                logging.error(str(e))
            print(e)
            solveLog.append(str(e) + '\n')
            return False

    def makeDeadImage(text):
        img = Image.new('RGB', (600, 200), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        myFont = ImageFont.truetype('FreeMono.ttf', 40)
        d.text((10, 10), text, fill=(100, 0, 0), font=myFont)
        arr = io.BytesIO()
        img.save(arr, format='JPEG')
        return arr.getvalue()

    cameraTry = 0
    print('solvethread', state)
    lastpictureTime = datetime.now()
    while True:
        lastsolveTime = datetime.now()


        if state is Mode.PAUSED or state is Mode.PLAYBACK:
            continue

        # solve this one selected image then switch state to playback
        if state is Mode.SOLVETHIS:

            print('solving skyStatus', skyStatusText, solveThisImage)
            copyfile(solveThisImage, os.path.join(solve_path, imageName))
            skyStatusText = 'Solving'
            #print("solving", solveThisImage)
            if skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['solver_type'] == 'solverTetra3':
                verboseSolveText = ""
                s = tetraSolve(Image.open(os.path.join(solve_path, imageName)))
                if s['RA'] != None:
                    ra = s['RA']
                    dec = s['Dec']
                    fov = s['FOV']
                    dur = (s['T_solve'])
                    result = "RA:%6.3lf    Dec:%6.3lf    FOV:%6.3lf time %6.3lf  secs" % (
                        ra/15, dec,  fov, dur)
                    skyStatusText = result
                else:
                    skyStatusText = str(s)
            else:

                if not solve(os.path.join(solve_path, imageName)) and skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['searchRadius'] > 0:
                    skyStatusText = 'Failed. Retrying with no position hint.'
                    if doDebug:
                        logging.warning("did not solve first try")
                    # try again but this time since the previous failed it will not use a starting guess possition
                    solve(os.path.join(solve_path, imageName))

            state = Mode.PLAYBACK
            solveCompleted = True

            continue

        else:  # live solving loop path
            #print("getting image in solve" )
            if cameraNotPresent:
                continue
            try: #why get image at top of loop
                debugLastState = "waiting for image"
                if doDebug:
                    logging.warning("waiting for image frame")
                solveReadyForImage = True
                frame = getImage()
                
                
            except Exception as e:
                cameraTry += 1
                if doDebug:
                    logging.error("no image after timeout retry count %s",cameraTry)
                if cameraTry > 10:

                    debugLastState = "no image after timeout"
                    saveImage(makeDeadImage("no camera. Restarting"))
                    print("camera failed" )
                    camera_Died = True
                    continue
                continue
            if doDebug:
                logging.warning("got frame %s",framecnt)
            if frame is None:
                print('frame is none')
                cameraTry += 1
                if cameraTry > 10:
                    if doDebug:
                        logging.error("empty frame after 10 retries")
                    debugLastState = "frame was none after 10 retries"
                    # saveImage(makeDeadImage("camera died. Restarting"))
                    print("camera died\nRestarting" )
                    camera_Died = True
                continue

            lastpictureTime = datetime.now()
        cameraTry = 0
        # if solving history one after the other in auto playback
        if (state is Mode.AUTOPLAYBACK):
            if testNdx == len(testFiles):
                state = Mode.PLAYBACK
                skyStatusText = "Complete."
                continue
            fn = testFiles[testNdx]

            solveThisImage = fn
            skyStatusText = "%d %s" % (testNdx, fn)
            solveLog.append(skyStatusText + '\n')
            testNdx += 1

            print ("image in auto playback  ", testNdx, fn)
     
            copyfile(fn, os.path.join(solve_path, imageName))

        #debug to fake camera image with a real star image
        #copyfile("static/history/11_01_22_23_47_15.jpeg", os.path.join(solve_path, imageName))
        if state is Mode.SOLVING or state is Mode.AUTOPLAYBACK :
            imagePath = os.path.join(solve_path, imageName)
            if skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['solver_type'] == 'solverTetra3':
                s = tetraSolve(frame)
                # print(str(s))
                if s['RA'] != None:
                    ra = s['RA']
                    dec = s['Dec']
                    fov = s['FOV']
                    dur = s['T_solve']
                    result = "RA:%6.3lf    Dec:%6.3lf     FOV:%6.3lf     %6.3lf secs" % (
                        ra/15, dec,  fov, dur)
                    skyStatusText = result
                else:
                    skyStatusText = str(s)
            else:
                if state is Mode.SOLVING:
                    skyStatusText = ""

                    cv2.imwrite(imagePath, frame)
                f = solve(imagePath)
                if f == False:
                    state = Mode.PLAYBACK
            solveCompleted = True
            continue

solveT = None
# print("config",skyConfig['solver'])
if skyConfig['solver']['startupSolveing']:
    print("should startup solver now")
    solveT = threading.Thread(target=solveThread)
    solveT.start()


def solve(fn, parms=[]):
    global doDebug, debugLastState, skyStatusText
    found = ''

    if doDebug:
        print("solving" )
        logging.warning("solving function")
    debugLastState = 'solving'
    global app, solving, maxTime, searchRaius, solveLog, ra, dec, searchEnable, solveStatus,\
        triggerSolutionDisplay, skyStatusText, lastObs, verboseSolveText
    startTime = datetime.now()
    solving = True
    solved = ''
    profile = skyConfig['solverProfiles'][skyConfig['solver']
                                          ['currentProfile']]
    fieldwidthParm = ''
    #print('fieldwidthMode', profile['FieldWidthMode'])
    if profile['FieldWidthMode'] == 'FieldWidthModeaPP':
        low = profile['aPPLoValue']
        high = profile['aPPHiValue']
        if high == '':
            field = ['--scale-units', 'arcsecperpix', '--scale-low', low]
        else:
            field = ['--scale-units', 'arcsecperpix',
                     '--scale-low', low, '--scale-high', high]
    elif profile['FieldWidthMode'] == 'FieldWidthModeField':
        low = profile['fieldLoValue']
        high = profile['fieldHiValue'].replace(' ', '')
        #print("highval", high)
        if high == '':
            field = ['--scale-units', 'degwidth', '--scale-low', low]
        else:
            field = ['--scale-units', 'degwidth',
                     '--scale-low', low, '--scale-high', high]
    else:
        field = []
    if profile['additionalParms'] != '':
        parmlist = profile['additionalParms'].split()
        parms = parms + parmlist
    parms = parms + field
    parms = parms + ['--cpulimit', str(profile['maxTime'])]
    if profile['searchRadius'] > 0 and ra != 0:
        parms = parms + ['--ra', str(ra), '--dec', str(dec),
                         '--radius', str(profile['searchRadius'])]
    #print('show stars', profile['showStars'])
    if not profile['showStars']:
        parms = parms + ['-p']
    parms = parms + ["--uniformize", "0", "--no-remove-lines", "--new-fits", "none",  "--pnm", "none", "--rdls",
                     "none"]
    cmd = ["solve-field", fn, "--depth", str(profile['solveDepth']), "--sigma", str(profile['solveSigma']),
           '--overwrite'] + parms
    solveLog.append(cmd)
    #print("\n\nsolving ", cmd)
    if skyConfig['observing']['verbose']:
        solveLog.append(' '.join(cmd) + '\n')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    ppa = ''
    starNames = {}
    duration = None
    radec = ''
    ra = 0
    dec = 0
    solveLog.append("solving:\n")
    #skyStatusText = skyStatusText + " solving"
    lastmessage = ''
    while not p.poll():
        stdoutdata = p.stdout.readline().decode(encoding='UTF-8')
        if stdoutdata:
            if stdoutdata == lastmessage:
                continue
            solveLog.append(stdoutdata)
            lastmessage = stdoutdata
            if 'simplexy: found' in stdoutdata:
                found = stdoutdata

                skyStatusText = stdoutdata
                #print("stdoutdata", stdoutdata)
            elif stdoutdata.startswith('Field center: (RA,Dec) = ('):
                solved = stdoutdata
                fields = solved.split()[-3:-1]
                #print ('f',fields)
                ra = fields[0][1:-1]
                dec = fields[1][0:-1]
                ra = float(fields[0][1:-1])
                dec = float(fields[1][0:-1])
                radec = "%s %6.6lf %6.6lf \n" % (
                    time.strftime('%H:%M:%S'), ra, dec)
                file1 = open(os.path.join(
                    solve_path, "radec.txt"), "w")  # write mode
                file1.write(radec)
                file1.close()
                if doDebug:
                    logging.warning('here is RAdec')
                    logging.warning(radec)
                stopTime = datetime.now()
                duration = stopTime - startTime
                #print ('duration', duration)
                skyStatusText = found + " solved. "+str(duration.total_seconds())+'secs'
            if stdoutdata and skyConfig['observing']['verbose']:

                #print("stdout", str(stdoutdata))
                #skyStatusText = skyStatusText + '.'
                if stdoutdata.startswith("Field 1: solved with"):
                    if doDebug:
                        ndx = stdoutdata.split("index-")[-1].strip()
                        print("index", ndx, stdoutdata )
                        usedIndexes[ndx] = usedIndexes.get(ndx, 0)+1
                        pp = pprint.pformat(usedIndexes)
                        print("Used indexes", pp )
                        solveLog.append('used indexes ' + pp + '\n')

                elif doDebug:
                    if stdoutdata.startswith('Field size'):
                        print("Field size", stdoutdata )

                        #solveStatus = found
                    elif stdoutdata.find('pixel scale') > 0:
                        computedPPa = stdoutdata.split("scale ")[1].rstrip()
                elif skyConfig['observing']['verbose']:
                    if 'The star' in stdoutdata:
                        stdoutdata = stdoutdata.replace(')', '')
                        con = stdoutdata[-4:-1]
                        if con not in starNames:
                            starNames[con] = 1

        else:
            break



    # create solved plot
    if solved and (state is Mode.PLAYBACK or skyConfig['observing']['verbose']):

        # Write-Overwrites
        file1 = open(os.path.join(solve_path, "radec.txt"), "w")  # write mode
        file1.write(radec)
        file1.close()
        cmdPlot = ['/usr/bin/plot-constellations', '-v', '-w', os.path.join(solve_path, 'cap.wcs'),
                   '-B', '-C', '-N', '-o', os.path.join(solve_path, 'cap-ngc.png')]
        p2 = subprocess.Popen(cmdPlot, stdout=subprocess.PIPE)
        ndx = 0
        stars = []
        while not p2.poll():
            if ndx > 1000:
                break
            ndx += 1
            stdoutdata = p2.stdout.readline().decode(encoding='UTF-8')
            if stdoutdata:
                stars.append(stdoutdata)
                #print(stdoutdata)

                if 'The star' in stdoutdata:
                    stdoutdata = stdoutdata.replace(')', '')
                    con = stdoutdata[-4:-1]
                    if con not in starNames:
                        starNames[con] = 1
        foundStars = ', '.join(stars).replace("\n", "")
        foundStars = foundStars.replace("The star", "")
        #solveLog.append(foundStars + "\n")
        constellations = ', '.join(starNames.keys())
        # copy the index over the ngc file so stars will display with the index triangle
        if profile['showStars']:
            os.remove(os.path.join(solve_path, 'cap-objs.png'))
            copyfile(os.path.join(solve_path, 'cap-indx.png'),
                     os.path.join(solve_path, 'cap-objs.png'))

        if skyConfig['observing']['savePosition'] and state is Mode.SOLVING:
            obsfile = open(os.path.join(solve_path, "obs.log"), "a+")
            obsfile.write(radec.rstrip() + " " + constellations + '\n')
            obsfile.close()
            #print ("wrote obs log")

        if skyConfig['observing']['saveImages']:
            saveimage = False
            if skyConfig['observing']['obsDelta'] > 0:
                #print ("checking delta")
                if lastObs:
                    old = lastObs.split(" ")
                    ra1 = math.radians(float(old[1]))
                    dec1 = math.radians(float(old[2]))
                    ra2 = math.radians(float(ra))
                    dec2 = math.radians(float(dec))

                    delta = math.degrees(math.acos(math.sin(
                        dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2)*math.cos(ra1 - ra2)))
                    #print("image delta", delta)
                    if delta > skyConfig['observing']['obsDelta']:
                        saveimage = True
                else:
                    saveimage = True
            if state is Mode.SOLVING and saveimage:
                lastObs = radec
                fn = datetime.now().strftime("%m_%d_%y_%H_%M_%S.jpg") 
                copyfile(os.path.join(solve_path, imageName),
                         os.path.join(solve_path, 'history', fn))

        if skyConfig['observing']['showSolution']:
            triggerSolutionDisplay = True
        stopTime = datetime.now()
        duration = stopTime - startTime
        skyStatusText = "solved "+str(duration.total_seconds()) + ' secs'
        verboseSolveText = foundStars
        solveLog.append(foundStars)
        if doDebug:
            logging.warning(skyStatusText)
            logging.warning(verboseSolveText)
            logging.warning(radec)
    if not solved:
        skyStatusText = skyStatusText + " Failed "

        ra = 0
        solveLog.append("Failed\n")
    solving = False
    solvestatestr = 'solved'
    if  not solved:
        solvestatestr = 'failed'
    if doDebug:
        logging.warning("solve finished with solve state %s", solvestatestr)
    return solved


def tetraSolve(img):
    global skyStatusText, solveLog, t3

    #solveLog.append("solving " + imageName + '\n')
    #img = Image.open(os.path.join(solve_path, imageName))

    image = np.asarray(img, dtype=np.float32)
    if image.ndim == 3:
        assert image.shape[2] in (1, 3), 'Colour image must have 1 or 3 colour channels'
        if image.shape[2] == 3:
            # Convert to greyscale
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            # Delete empty dimension
            image = image.squeeze(axis=2)
    else:
        assert image.ndim == 2, 'Image must be 2D or 3D array'
    print('solving', imageName)
    profile = skyConfig['solverProfiles'][skyConfig['solver']
                                          ['currentProfile']]
    t0 = precision_timestamp()
    (width, height) = image.shape
    centroids = get_centroids_from_image(image)
    print('centroids', len(centroids), precision_timestamp() - t0)
    trimmed_centroids = centroids[:30]
    solution = t3.solve_from_centroids(
                        trimmed_centroids, (height, width),
                        return_matches=True, solve_timeout=5000)
                            # Don't clutter printed solution with these fields.
    solution.pop('matched_centroids', None)
    solution.pop('matched_stars', None)
    solution.pop('matched_catID', None)
    solution.pop('pattern_centroids', None)
    solution.pop('epoch_equinox', None)
    solution.pop('epoch_proper_motion', None)
    solution.pop('cache_hit_fraction', None)

    if solution['RA'] == None:
        return solution
    tdel = precision_timestamp() - t0
 
    print('solution',tdel,solution)
    solution['T_solve'] = tdel
    radec = "%6.2lf %6.6lf  %6.6lf \n" % (tdel, solution['RA'], solution['Dec'])
    solveLog.append(str(solution) + '\n')
    file1 = open(os.path.join(solve_path, "radec.txt"), "w")  # write mode
    file1.write(radec)
    file1.close()
    skyStatusText = str(solution['RA'])
    return solution


app = Flask(__name__)
doDebug = False
skyStatusText = 'Initilizing Camera'


@app.route("/StarHistory", methods=['GET','POST'])

def showStarHistoryPage():
    return render_template('starHistory.html')


@app.route("/", methods=['GET', 'POST'])
def index():
    global skyCam, cameraNotPresent, skyStatusText, solveT, verboseSolveText
    verboseSolveText = ""
    shutterValues = ['.001', '.002', '.005', '.01', '.02', '.05', '.1', '.15', '.2',
                     '.5', '.7', '.9', '1', '2.', '3', '4', '5', '10']

    currentShutter = skyConfig['camera']['shutter']
    skyFrameValues = ['400x300', '640x480', '800x600', '1024x768',
                      '1280x960', '1920x1440', '2000x1000', '2000x1500']
    isoValues = ['10', '50', '100', '200', '400', '800', '1000', '2000','4000','8000','16000']
    formatValues = ['jpg', 'png']
    currentISO = skyConfig['camera']['ISO']
    currentFrameValue = skyConfig['camera']['frame']
    print('ISO',currentISO)
    solveParams = {'PPA': 27, 'FieldWidth': 14, 'Timeout': 34,
                   'Sigma': 9, 'Depth': 20, 'SearchRadius': 10}
    if cameraNotPresent:
        skyStatusText = 'camera not connected or enabled.  Demo mode and replay mode will still work however.'
        print('camera was not found')
        #print('current profile', skyConfig['solver'])
    if not solveT:  # start up solver if not already running.
        solveT = threading.Thread(target=solveThread)
        solveT.start()
    return render_template('template.html', shutterData=shutterValues, skyFrameData=skyFrameValues, skyFormatData=formatValues,
                           skyIsoData=isoValues, profiles=skyConfig['solverProfiles'],
                            shutterValue = currentShutter, ISOValue = currentISO, frameValue = currentFrameValue,
                            startup=skyConfig['solver']['startupSolveing'], solveP=skyConfig['solver']['currentProfile'], obsParms=skyConfig['observing'], cameraParms=skyConfig['camera'])


@app.route('/pause', methods=['POST'])
def pause():
    global skyStatusText, skyCam, state
    if skyCam:
        skyCam.pause()
    state = Mode.PAUSED

    skyStatusText = 'Paused'
    return Response(skyStatusText)


@app.route('/Align', methods=['POST'])
def Align():
    global skyStatusText, skyCam, state
    if skyCam:
        skyCam.resume()

        state = Mode.ALIGN
        skyStatusText = 'Align Mode'
    return Response(skyStatusText)


@app.route('/saveCurrent', methods=['POST'])
def saveCurrent():
    global skyStatusText, imageName
    print("save current image" )
    fn = datetime.now().strftime("%m_%d_%y_%H_%M_%S.") + \
        skyConfig['camera']['format']
    copyfile(os.path.join(solve_path, imageName),
             os.path.join(solve_path, 'history', fn))

    skyStatusText = 'saved'
    return Response(skyStatusText)


@app.route('/Solve', methods=['POST'])
def Solving():
    global skyStatusText, state, skyCam
    if skyCam:
        skyCam.resume()
    if state is Mode.PLAYBACK:
        state = Mode.AUTOPLAYBACK
        skyStatusText = "Auto playback"
    elif state is Mode.AUTOPLAYBACK:
        state = Mode.PLAYBACK
        skyStatusText = "Manual playback"
    else:
        state = Mode.SOLVING
        skyStatusText = 'Solving Mode'
    return Response(skyStatusText)





@app.route('/setISO/<value>', methods=['POST'])
def setISO(value):

    print("setting iso", value)

    skyCam.setISO(int(value))

    skyConfig['camera']['ISO'] = value
    saveConfig()

    return Response(status=204)


@app.route('/setFrame/<value>', methods=['POST'])
def setFrame(value):
    skyCam.setResolution(value)
    skyConfig['camera']['frame'] = value
    saveConfig()
    return Response(status=204)


@app.route('/setFormat/<value>', methods=['POST'])
def setFormat(value):
    global skyStatusText, imageName
    skyStatusText = "changing Image Format to " + value
    imageName = 'cap.'+value
    skyCam.setFormat(value)
    skyConfig['camera']['format'] = value
    saveConfig()

    return Response(status=204)





@app.route('/setShutter/<value>', methods=['POST'])
def setShutter(value):
    global skyStatusText

    skyCam.setShutter(float(value))
    skyConfig['camera']['shutter'] = value

    saveConfig()

    return Response(status=204)


@app.route('/showSolution/<value>', methods=['POSt'])
def setShowSolutions(value):
    global skyConfig
    print('show solutions', value)
    if (value == '1'):
        skyConfig['observing']['showSolution'] = value == '1'
    saveConfig()
    return Response(status=204)


@app.route('/saveImages/<value>', methods=['POST'])
def saveSolvedImage(value):
    global skyConfig
    print('save images ', value)
    skyConfig['observing']['saveImages'] = value == '1'
    saveConfig()
    print("config", skyConfig)
    return Response(status=204)


@app.route('/verbose/<value>', methods=['POST'])
def verbose(value):
    global skyConfig
    print("verbose", value)
    skyConfig['observing']['verbose'] = (value == 'true')
    saveConfig()
    print("config verbose", skyConfig['observing']['verbose'])
    return Response(status=204)


@app.route('/clearObsLog', methods=['POST'])
def clearObsLog():
    global ndx, obsList
    ndx = 0
    obsList.clear()
    with open(os.path.join(solve_path, 'obs.log'), 'w') as infile:
        pass  # this will write an empty file erasing the previous contents
    return Response("observing log cleared")


setTimeDate = True


@app.route('/lastVerboseSolve', methods=['post'])
def verboseSolveUpdate():
    global verboseSolveText
    return Response(verboseSolveText)

@app.route("/solveLog")
def streamSolveLog():
    global solveLog
    #print (len(solveLog))
    str = ''
    while True:
        try:
            s = solveLog.popleft()
            str += s
        except:
            break

    return Response(str)

@app.route('/skyStatus', methods=['post'])
def skyStatus():
    global skyStatusText, setTimeDate
    if setTimeDate:
        setTimeDate = False
        t = float(request.form['time'])/1000
        try:
            time.clock_settime(time.CLOCK_REALTIME, t)  
        except BaseException as e:
            print("could not set clock", e )
            
    resp = copy.deepcopy(skyStatusText)


    
    #skyStatusText = ''
    return Response(resp)

 
@app.route('/Focus', methods=['post'])
def Focus():
    global focusStd
    return Response(focusStd)

lastmode = state
def gen():
    global skyStatusText, solveT, testNdx,  triggerSolutionDisplay,\
            doDebug, state, solveCurrent,  framecnt, lastDisplayedFile,lastmode,\
                GUIReadyForImage, GUIImage, imageName
    # Video streaming generator function.
    print("gen called")
    lastImageTime = datetime.now()
    yield (b'\r\n--framex\r\n' b'Content-Type: image/jpeg\r\n\r\n')
    while True:
        frame = None
        if state is Mode.PAUSED:
            time.sleep(1)
            continue

        if (state != Mode.ALIGN) and (state != Mode.SOLVING):

            if lastDisplayedFile == solveThisImage:
                continue
            print("history",lastDisplayedFile, solveThisImage)
            lastDisplayedFile = solveThisImage
            with open(solveThisImage, 'rb') as infile:
                frame = infile.read()


        else:
            GUIReadyForImage = True
            frame = cv2.imencode('.jpeg', getImage())[1].tobytes()

        if doDebug:
            logging.warning("frame sent to GUI %d", framecnt)
            if state is Mode.SOLVING:
                framecnt = framecnt + 1
            
        if state is Mode.ALIGN:
            framecnt = framecnt + 1
            skyStatusText = "frame %d" % (framecnt)
        


        # this send the image and also the header for the next image
        yield (frame + b'\r\n--framex\r\n' b'Content-Type: image/jpeg\r\n\r\n')

        if state is Mode.SOLVING:
            solveCurrent = True


@app.route('/deleteProfile/<value>', methods=['POST'])
def deleteProfile(value):
    print('delete profile', value, ">", skyConfig['solverProfiles'].keys())
    if value in skyConfig['solverProfiles']:
        del skyConfig['solverProfiles'][value]

    return json.dumps(skyConfig['solverProfiles'])


@app.route('/applySolve2', methods=['GET', 'POST'])
def apply():
    print("submitted values2", request.form.values)

    req = request.form
    cur = req.get("currentProfile").strip()
    if cur not in skyConfig['solverProfiles']:
        newProfile = copy.deepcopy(skyConfig['solverProfiles']['default'])
        newProfile['name'] = cur
        skyConfig['solverProfiles'][cur] = newProfile

    skyConfig['solver']['currentProfile'] = cur

    skyConfig['solver']['startupSolveing'] = bool(req.get('startupType'))
    profile = skyConfig['solverProfiles'][cur]

    #print('\n\nProfile', profile)
    mode = profile['FieldWidthMode'] = req.get("FieldWidthMode")
    profile['FieldWidthMode'] = mode

    profile['name'] = cur
    profile['fieldLoValue'] = req.get("fieldLoValue")
    profile['fieldHiValue'] = req.get("fieldHiValue")
    profile['aPPLoValue'] = req.get("aPPLowValue")
    profile['aPPHiValue'] = req.get("aPPHiValue")
    if req.get("searchRadius") != "":
        profile['searchRadius'] = float(req.get("searchRadius"))
    else:
        profile['searchRadius'] = 0
    profile['solveSigma'] = int(req.get("solveSigma"))
    profile['solveDepth'] = int(req.get("solveDepth"))
    profile['solveVerbose'] = bool(req.get("solveVerbose"))
    profile['showStars'] = bool(req.get("showStars"))
    profile['verbose'] = bool(req.get("verbose"))
    profile['maxTime'] = float(req.get('CPUTimeout'))
    profile['additionalParms'] = req.get('additionalParms')
    #print("curprofile", profile)
    saveConfig()
    print('\n\n\nskyconfig', json.dumps(
        skyConfig['solverProfiles'][cur], indent=4))

    # print (request.form.['submit_button'])

    return json.dumps(skyConfig['solverProfiles'])


@app.route('/demoMode', methods=['POST'])
def demoMode():
    global testMode, testFiles, testNdx,   solveLog, state, solveThisImage

    skyStatusText = 'Demo images will be set for playback'
    state = Mode.PLAYBACK
    testFiles = [demo_path + '/' + fn for fn in os.listdir(
        demo_path) if any(fn.endswith(ext) for ext in ['jpg', 'png'])]
    testFiles.sort(key=os.path.getmtime)
    testNdx = 0

    print("demo files len", len(testFiles))
    time.sleep(2)

    setupImageFromFile()



    return Response(skyStatusText)


def setupImageFromFile():
    global solveThisImage,   skyStatusText

    solveThisImage = testFiles[testNdx]
    skyStatusText = "%d %s" % (testNdx, testFiles[testNdx])


def findHistoryFiles():
    global saveLog, skyStatusText, testFiles, testNdx,  solveThisImage
    print("finding files")
    x = datetime.now() + timedelta(seconds=2)
    testFiles = [history_path + '/' + fn for fn in os.listdir(
        history_path) if any(fn.endswith(ext) for ext in ['jpg', 'jpeg', 'png'])]
    testFiles.sort(key=os.path.getmtime)


    print("test files len", len(testFiles))


    while datetime.now() < x:
        time.sleep(1)
    print("gather done")
    if len(testFiles) == 0:
        skyStatusText = "no image files in hisotry"
    else:
        skyStatusText = 'PLAYBACK mode ' + \
            str(len(testFiles)) + " images found."
        solveThisImage = testFiles[0]
        print(skyStatusText)
        time.sleep(3)  # sleep to let status update number of images found
        setupImageFromFile()

    print(skyStatusText)
    # change status to display the first file name after 3 seconds
    th = threading.Thread(target=delayedStatus, args=(5, solveThisImage))
    th.start()


def reboot3():
    time.sleep(3)
    os.system('sudo reboot')


@app.route('/shutdown', methods=['post'])
def shutdown():
    global skyStatusText
    skyStatusText = "shutting down.  Good Bye"
    th = threading.Thread(target=shutThread)
    th.start()


@app.route('/reboot', methods=['POST'])
def reboot():
    global skyStatusText, state
    state = Mode.PAUSED
    th = threading.Thread(target=reboot3)
    th.start()
    skyStatusText = "reboot in 3 seconds goodbye. You will need to reload this page after about 3 minutes"
    return Response(skyStatusText)


def restartThread():
    print("restarting thread waiting for 5 seconds" )
    time.sleep(5)
    os.system('./restartsky.sh')


@app.route('/restartc', methods=['POST'])
def restartc():
    global skyStatusText, skyCam, state

    state = Mode.PAUSED
    th = threading.Thread(target=restartThread)
    th.start()
    skyStatusText = 'restarting. You will need to Reload this page in about 30 seconds'
    return Response(skyStatusText)


@app.route('/testMode', methods=['POST'])
def toggletestMode():
    global testMode, testFiles, testNdx,   solveLog, state, solveThisImage, skyStatusText

    state = Mode.PLAYBACK
    print('will find files')
    skyStatusText = "Gathering History files"
    th = threading.Thread(target=findHistoryFiles)
    th.start()

    return Response(skyStatusText)

measureHtmlStack = []
measureStackLock = threading.Lock()
allDone = False
# measure transparency of current image
import traceback


readytoSend = False
message = ''
statusCols = ['','']

def sendStatus(msg, col=1):
    global readytoSend, message
    statusCols[col-1] = msg
    while readytoSend:
        pass
    message = ' '.join(statusCols)
    readytoSend = True
    


    
@app.route('/showImage')
def processImageX():
    name = request.args.get('fn')
    print("processImageX",name)

    html = '<!DOCTYPE html><html><body  style=" background-color: rgb(0, 0, 0)" ><div style="color:#000000" ><p style="color:#000000">'\
    '<img style = "filter:brightness(500%%); color:#000000" src=" ./static/history/%s">'\
    '</p></div></body></html>'%(name)

    return(html)


@app.route('/sqDelete', methods=['POST'])
def deletecurrentHistoryImage():
    global testNdx, testFiles, state
    if state != Mode.PLAYBACK:
        return('%d %s'%('Replay Images must be selected first', ' '))
    fn = testFiles[testNdx]
    testFiles.remove(fn)
    print("deleting history file- ", testNdx, fn)
    os.remove(fn)

    setupImageFromFile()
    
    return Response("deleted");


@app.route('/nextImage', methods=['POST'])
def nextImagex():
    global testNdx
    testNdx += 1
    if testNdx >= len(testFiles):
        testNdx = 0

    setupImageFromFile()
    return Response(skyStatusText)


@app.route('/prevImage', methods=['POST'])
def prevImagex():
    global testNdx
    if (testNdx > 0):
        testNdx -= 1
    else:
        testNdx = len(testFiles)-1
    print('prev pressed')
    setupImageFromFile()
    return Response(skyStatusText)


@app.route('/startup/<value>', methods=['POST'])
def startup(value):
    if value == 'true':
        skyConfig['solver']['startupSolveing'] = True
    else:
        skyConfig['solver']['startupSolveing'] = False
    saveConfig()

    return Response(status=204)


@app.route('/historyNdx', methods=['POST'])
def historyNdx():
    global testNdx, solveThisImage, state
    print("submitted values1", request.form.values)
    if state is Mode.PLAYBACK:
        req = request.form
        testNdx = int(req.get("hNdx"))


    setupImageFromFile()
    return Response(status=205)

@app.route('/Debug', methods= ['POST'])
def debugcommands():
    global doDebug
    doDebug = request.form.get("enableDebug") == "on"
    if (doDebug):
        sys.stdout = x = ListStream()
    else:
        sys.stdout = sys.__stdout__
    print("debug",doDebug, request.form.get("enableDebug"))
    return Response(status=205)

@app.route('/Observe', methods=['POST'])
def setObserveParams():
    skyConfig['observing']['saveImages'] = request.form.get("saveImages")
    skyConfig['observing']['obsDelta'] = float(request.form.get('deltaDiff'))
    skyConfig['observing']['savePosition'] = bool(request.form.get('SaveOBS'))

    saveConfig()

    skyStatusText = "observing session parameters received."
    return Response(status=205)


@app.route('/startObs', methods=['POST'])
def startObs():
    global ndx, obsList
    ndx = 0
    with open(os.path.join(solve_path, 'obs.log'), 'r') as infile:
        obsList = infile.readlines()

    return updateRADEC()


def updateRADEC():
    global ndx, obsList
    radec = obsList[ndx]
    file1 = open(os.path.join(solve_path, "radec.txt"), "w")  # write mode
    file1.write(radec)
    file1.close()
    return Response(radec.rstrip())


@app.route('/nextObs', methods=['POST'])
def nextObs():
    global ndx, obsList
    ndx += 1
    if ndx > len(obsList):
        ndx -= 1
    return updateRADEC()


@app.route('/prevObs', methods=['POST'])
def prevObs():
    global ndx, obsList
    ndx -= 1
    if ndx < 0:
        ndx = 0
    return updateRADEC()


@app.route('/solveThis', methods=['POSt'])
def solveThis():
    global solveCurrent, state
    solveCurrent = True
    state = Mode.SOLVETHIS
    skyStatusText = "Solving"
    return Response(skyStatusText)


# Zip the files from given directory that matches the filter
def zipFilesInDir(dirName, zipFileName):
    # create a ZipFile object
    with ZipFile(zipFileName, 'w') as zipObj:
        valid_images = [".jpg", ".png", ".jpeg"]
        imgs = []
        for f in os.listdir(dirName):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            imgs.append(f)
        print("images", imgs)
        for filename in imgs:
            # create complete filepath of file in directory
            filePath = os.path.join(solve_path, 'history', filename)
            # Add file to zip
            zipObj.write(filePath, filename)


@app.route('/clearImages', methods=['POST'])
def clearImages():
    global solve_path
    valid_images = [".jpg", ".png", ".jpeg"]
    imgs = []
    for f in os.listdir(os.path.join(solve_path, 'history')):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        os.remove(os.path.join(solve_path, 'history', f))
    return Response("images deleted")

@app.route('/downloadImage', methods=['GET'])
def downloadImage():
    global  solveThisImage

    if (state != Mode.ALIGN) and (state != Mode.SOLVING):
        print("solve this image was", solveThisImage)
        return send_file(solveThisImage, as_attachment=True)

    else:
        frame = skyCam.get_frame()

        fn = os.path.join(solve_path, "current.jpg")
        cv2.imwrite(fn, frame)

        return send_file(fn, as_attachment=True)

@app.route('/zip', methods=['GET'])
def zipImages():
    global solve_path, skyStatusText
    print("download zip file")
    skyStatusText = "zipping images"
    solveLog.append("zipping images into history.zip\n")
    zipFilesInDir(os.path.join(solve_path, 'history'),
                  os.path.join(solve_path, 'history', 'history.zip'))

    skyStatusText = "history.zip file is being sent."
@app.route('/video_feed')
def video_feed():
    r = Response(gen(),
                 mimetype='multipart/x-mixed-replace; boundary=framex')

    return (r)


if __name__ == '__main__':
    print("working dir", os.getcwd())
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
