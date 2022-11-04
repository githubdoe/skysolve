from subprocess import call
from zipfile import ZipFile
from attr import get_run_validators
from flask import Flask, render_template, request, Response, send_file
import time
from flask.wrappers import Request
import pprint

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
from tetra3 import Tetra3
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
import Quality

import glob

import matplotlib.pyplot as plt
from scipy import stats

print("argssss", sys.argv, len(sys.argv))
print('user', getpass.getuser())
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.IN, pull_up_down=GPIO.PUD_UP)
initGPIO7 = GPIO.input(7)
print("gpio", initGPIO7)
try:
    os.system('systemctl restart encodertoSkySafari.service ')
except BaseException as e:
    print("did not start encoder", e, flush=True)

usedIndexes = {}

# Create instance and load default_database (built with max_fov=12 and the rest as default)
t3 = None
if t3 == None:
    t3 = Tetra3('default_database')


class Mode(Enum):
    PAUSED = auto()
    ALIGN = auto()
    SOLVING = auto()
    PLAYBACK = auto()
    SOLVETHIS = auto()
    # playback and sovlve history images without user clicking on each.
    AUTOPLAYBACK = auto()


class LimitedLengthList(list):
    def __init__(self, seq=(), length=math.inf):
        self.length = length

        if len(seq) > length:
            raise ValueError("Argument seq has too many items")

        super(LimitedLengthList, self).__init__(seq)

    def append(self, item):
        if len(self) < self.length:
            super(LimitedLengthList, self).append(item)

        else:
            super(LimitedLengthList, self).__init__(
                super(LimitedLengthList, self)[self.length/2:])
            super(LimitedLengthList, self).append(item)


app = 30
solving = False
maxTime = 50
searchEnable = False
searchRadius = 90
solveLog = LimitedLengthList(length=1000)
ra = 0
dec = 0
solveStatus = ''
computedPPa = ''
frameStack = []  # holds images and the time it was taken
frameStackLock = threading.Lock()


focusStd = ''
state = Mode.ALIGN

testNdx = 0
testFiles = []
root_path = os.getcwd()
if len(sys.argv) == 2:
    root_path = sys.argv[1]


solve_path = os.path.join(root_path, 'static')
history_path = os.path.join(solve_path, 'history')
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


def saveConfig():
    with open('skyConfig.json', 'w') as f:

        json.dump(skyConfig, f, indent=4)


# saveConfig()
print('cwd', os.getcwd())


with open(os.path.join(root_path, 'skyConfig.json')) as f:
    skyConfig = json.load(f)
print(json.dumps(skyConfig['solverProfiles']['25FL'], indent=4))
print(json.dumps(skyConfig['observing'], indent=4))
print(json.dumps(skyConfig['camera'], indent=4), flush=True)

imageName = 'cap.'+skyConfig['camera']['format']

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
        print('creating cam')
        try:
            skyCam = skyCamera(delayedStatus, shutter=int(
                1000000 * float(skyConfig['camera']['shutter'])),
                format=skyConfig['camera']['format'],
                resolution=skyConfig['camera']['frame'])
            cameraNotPresent = False
            if skyConfig['solver']['startupSolveing']:
                print("startup in solving")
                state = Mode.SOLVING
            startFrame = skyCam.get_frame()
            if startFrame is None:
                print("camera did not seem to start")
            print("camera started and frame received",
                  cameraNotPresent, flush=True)
        except Exception as e:
            print(e)
            cameraNotPresent = True
            skyStatusText = 'camera not connected or enabled.  Demo mode and replay mode will still work however.'


setupCamera()

framecnt = 0


lastsolveTime = datetime.now()
justStarted = True
camera_Died = False
solveCompleted = False


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
    global skyStatusText, focusStd, solveCurrent, state, skyCam, frameStack, frameStackLock, testNdx, camera_Died,\
        solveLog, solveCompleted

    # save the image to be solved in the file system and on the stack for the gen() routine to give to the client browser
    def saveImage(frame):
        global frameStack, frameStackLock

        with open(os.path.join(solve_path, imageName), "wb") as f:
            try:
                f.write(frame)
                frameStackLock.acquire()
                # save to stack so gen can send it to the client browser
                frameStack.append(
                    (os.path.join(solve_path, imageName), datetime.now()))
                frameStackLock.release()
                return True
            except Exception as e:
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
            print("solving", solveThisImage)
            if skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['solver_type'] == 'solverTetra3':
                verboseSolveText = ""
                s = tetraSolve(os.path.join(solve_path, imageName))
                if s['RA'] != None:
                    ra = s['RA']
                    dec = s['Dec']
                    fov = s['FOV']
                    dur = (s['T_solve']+s['T_extract'])/1000
                    result = "RA:%6.3lf    Dec:%6.3lf    FOV:%6.3lf %6.3lf      secs" % (
                        ra/15, dec,  fov, dur)
                    skyStatusText = result
                else:
                    skyStatusText = str(s)
            else:

                if not solve(os.path.join(solve_path, imageName)) and skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['searchRadius'] > 0:
                    skyStatusText = 'Failed. Retrying with no position hint.'
                    # try again but this time since the previous failed it will not use a starting guess possition
                    solve(os.path.join(solve_path, imageName))

            state = Mode.PLAYBACK
            solveCompleted = True
            continue

        else:  # live solving loop path
            #print("getting image in solve", flush=True)
            if cameraNotPresent:
                continue
            try:
                frame = skyCam.get_frame()
            except Exception as e:
                cameraTry += 1

                if cameraTry > 10:
                    saveImage(makeDeadImage("no camera. Restarting"))
                    print("camera failed", flush=True)
                    camera_Died = True
                    continue
                continue

            if frame is None:
                cameraTry += 1
                if cameraTry > 10:
                    saveImage(makeDeadImage("camera died. Restarting"))
                    print("camera died\nRestarting", flush=True)
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
            skyStatusText = "%d %s" % (testNdx, fn)
            solveLog.append(skyStatusText + '\n')
            testNdx += 1

            #print ("image", testNdx, fn)

            with open(fn, 'rb') as infile:
                frame = infile.read()

        saveImage(frame)
        #debug to fake camera image with a real star image
        #copyfile("static/history/11_01_22_23_47_15.jpeg", os.path.join(solve_path, imageName))
        if state is Mode.SOLVING or state is Mode.AUTOPLAYBACK:
            if skyConfig['solverProfiles'][skyConfig['solver']['currentProfile']]['solver_type'] == 'solverTetra3':
                s = tetraSolve(os.path.join(solve_path, imageName))
                # print(str(s))
                if s['RA'] != None:
                    ra = s['RA']
                    dec = s['Dec']
                    fov = s['FOV']
                    dur = (s['T_solve']+s['T_extract'])/1000
                    result = "RA:%6.3lf    Dec:%6.3lf     FOV:%6.3lf     %6.3lf secs" % (
                        ra/15, dec,  fov, dur)
                    skyStatusText = result
                else:
                    skyStatusText = str(s)
            else:
                if state is Mode.SOLVING:
                    skyStatusText = ""
                f = solve(os.path.join(solve_path, imageName))
                if f == False:
                    state = Mode.PLAYBACK
            solveCompleted = True
            continue

        # else measaure contrast for focus bar
        try:
            img = Image.open(os.path.join(solve_path, imageName)).convert("L")
            imgarr = np.array(img)
            avg = imgarr.mean()
            imax = imgarr.max()
            if avg == 0:
                avg = 1

            contrast = (imax - avg)/avg

            focusStd = f"{contrast:.2f}"

        except Exception as e:
            print(e)


solveT = None
# print("config",skyConfig['solver'])
if skyConfig['solver']['startupSolveing']:
    print("should startup solver now")
    solveT = threading.Thread(target=solveThread)
    solveT.start()



def solve(fn, parms=[]):
    print("solving", flush=True)

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

    #print("\n\nsolving ", cmd)
    if skyConfig['observing']['verbose']:
        solveLog.append(' '.join(cmd) + '\n')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    solveLog.clear()
    ppa = ''
    starNames = {}
    duration = None
    radec = ''
    ra = 0
    dec = 0
    solveLog.append("solving:\n")
    skyStatusText = skyStatusText + " solving"
    lastmessage = ''
    while not p.poll():
        stdoutdata = p.stdout.readline().decode(encoding='UTF-8')
        if stdoutdata:
            if stdoutdata == lastmessage:
                continue
            lastmessage = stdoutdata
            if 'simplexy: found' in stdoutdata:
                skyStatusText = stdoutdata
                print("stdoutdata", stdoutdata)
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
                stopTime = datetime.now()
                duration = stopTime - startTime
                #print ('duration', duration)
                skyStatusText = skyStatusText + " solved "+str(duration)+'secs'
            if stdoutdata and skyConfig['observing']['verbose']:
                solveLog.append(stdoutdata)
                print("stdout", str(stdoutdata))
                skyStatusText = skyStatusText + '.'
                if stdoutdata.startswith("Field 1: solved with"):
                    ndx = stdoutdata.split("index-")[-1].strip()
                    print("index", ndx, stdoutdata, flush=True)
                    usedIndexes[ndx] = usedIndexes.get(ndx, 0)+1
                    pp = pprint.pformat(usedIndexes)
                    print("Used indexes", pp, flush=True)
                    solveLog.append('used indexes ' + pp + '\n')

                elif stdoutdata.startswith('Field size'):
                    print("Field size", stdoutdata, flush=True)
                    solveLog.append(stdoutdata)
                    solveStatus += (". " + stdoutdata.split(":")[1].rstrip())
                elif stdoutdata.find('pixel scale') > 0:
                    computedPPa = stdoutdata.split("scale ")[1].rstrip()
                elif 'The star' in stdoutdata:
                    stdoutdata = stdoutdata.replace(')', '')
                    con = stdoutdata[-4:-1]
                    if con not in starNames:
                        starNames[con] = 1

        else:
            break

    solveStatus += ". scale " + ppa

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
                print(stdoutdata)

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
                fn = datetime.now().strftime("%m_%d_%y_%H_%M_%S.") + \
                    skyConfig['camera']['format']
                copyfile(os.path.join(solve_path, imageName),
                         os.path.join(solve_path, 'history', fn))

        if skyConfig['observing']['showSolution']:
            triggerSolutionDisplay = True
        stopTime = datetime.now()
        duration = stopTime - startTime
        skyStatusText = "solved "+str(duration) + ' secs'
        verboseSolveText = foundStars
    if not solved:
        skyStatusText = skyStatusText + " Failed"
        ra = 0
        solveLog.append("Failed\n")
    solving = False
    return solved


def tetraSolve(imageName):
    global skyStatusText, solveLog, t3

    solveLog.append("solving " + imageName + '\n')
    img = Image.open(os.path.join(solve_path, imageName))
    #print('solving', imageName)
    profile = skyConfig['solverProfiles'][skyConfig['solver']
                                          ['currentProfile']]

    solved = t3.solve_from_image(
        img, fov_estimate=float(profile['fieldLoValue']))

    # print(str(solved),profile['fieldLoValue'],flush=True)
    if solved['RA'] == None:
        return solved
    radec = "%s %6.6lf %6.6lf \n" % (time.strftime(
        '%H:%M:%S'), solved['RA'], solved['Dec'])
    solveLog.append(str(solved) + '\n')
    file1 = open(os.path.join(solve_path, "radec.txt"), "w")  # write mode
    file1.write(radec)
    file1.close()
    skyStatusText = str(solved['RA'])
    return solved


app = Flask(__name__)

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
    skyFrameValues = ['400x300', '640x480', '800x600', '1024x768',
                      '1280x960', '1920x1440', '2000x1000', '2000x1500']
    isoValues = ['100', '200', '400', '800']
    formatValues = ['jpeg', 'png']
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
                           skyIsoData=isoValues, profiles=skyConfig['solverProfiles'], startup=skyConfig['solver']['startupSolveing'], solveP=skyConfig['solver']['currentProfile'], obsParms=skyConfig['observing'], cameraParms=skyConfig['camera'])

#


@app.route("/solveLog")
def streamSolveLog():
    global solveLog

    def generate():
        global solveLog
        while True:
            #print (len(solveLog))
            if len(solveLog) > 0:
                str = solveLog.pop(0)
                #print ("log", str)
                yield str

    return app.response_class(generate(), mimetype="text/plain")


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
    print("save current image", flush=True)
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


@app.route('/setISO', methods=['POST'])
def setISOx():
    print("setISO FORM", request.form.values)
    global solveLog, skyStatusText, isoglobal
    value = request.form.get('setISO')
    solveLog.append("ISO changing will take 10 seconds to stabilize gain.\n")
    delayedStatus(2, "changing to ISO " + value)
    isoglobal = value
    skyCam.setISO(int(value))

    skyConfig['camera']['ISO'] = value
    saveConfig()
    return Response(status=204)


@app.route('/setISO/<value>', methods=['POST', 'GET'])
def setISO(value):

    print("setting iso", value)
    global solveLog, skyStatusText, isoglobal
    solveLog.append("ISO changing will take 10 seconds to stabilize gain.\n")
    delayedStatus(2, "changing to ISO " + value)
    isoglobal = value
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


@app.route('/setShutter', methods=['POST'])
def setShutterx():
    global skyStatusText
    value = request.form.get('setShutter')
    print("shutter value", value)
    skyCam.setShutter(int(1000000 * float(value)))
    skyConfig['camera']['shutter'] = value
    delayedStatus(2, "Setting shutter to "+str(value) +
                  " may take about 10 seconds.")
    saveConfig()

    return Response(status=204)


@app.route('/setShutter/<value>', methods=['POST'])
def setShutter(value):
    global skyStatusText
    print("shutter value", value)
    skyCam.setShutter(int(1000000 * float(value)))
    skyConfig['camera']['shutter'] = value
    delayedStatus(2, "Setting shutter to "+str(value) +
                  " may take about 10 seconds.")
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


@app.route('/skyStatus', methods=['post'])
def skyStatus():
    global skyStatusText, setTimeDate
    if setTimeDate:
        setTimeDate = False
        t = float(request.form['time'])/1000
        time.clock_settime(time.CLOCK_REALTIME, t)
    return Response(skyStatusText)


@app.route('/Focus', methods=['post'])
def Focus():
    global focusStd
    return Response(focusStd)


def gen():
    global skyStatusText, solveT, testNdx,  triggerSolutionDisplay, testMode, state, solveCurrent, frameStack, frameStackLock, framecnt, tmr
    # Video streaming generator function.

    lastImageTime = datetime.now()
    yield (b'\r\n--framex\r\n' b'Content-Type: image/jpeg\r\n\r\n')
    while True:
        if len(frameStack) == 0:
            continue

        frameStackLock.acquire()
        if len(frameStack) == 0:

            frameStackLock.release()
            continue

        lastImageTime = frameStack[-1][1]

        fn = frameStack[-1][0]

        with open(fn, 'rb') as infile:
            frame = infile.read()
        frameStack.clear()
        frameStackLock.release()

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
    global testMode, testFiles, testNdx, frameStack,  solveLog, state, solveThisImage

    skyStatusText = 'Demo images will be set for playback'
    state = Mode.PLAYBACK
    testFiles = [demo_path + '/' + fn for fn in os.listdir(
        demo_path) if any(fn.endswith(ext) for ext in ['jpg', 'png'])]
    testFiles.sort(key=os.path.getmtime)
    testNdx = 0

    print("demo files len", len(testFiles))
    time.sleep(2)
    frameStack.clear()
    setupImageFromFile()

    solveLog = [x + '\n' for x in testFiles]

    return Response(skyStatusText)


def setupImageFromFile():
    global solveThisImage, frameStackLock, frameStack, skyStatusText

    solveThisImage = testFiles[testNdx]

    frameStackLock.acquire()
    frameStack.clear()

    frameStack.append((solveThisImage, datetime.now()))
    frameStackLock.release()

    skyStatusText = "%d %s" % (testNdx, testFiles[testNdx])


def findHistoryFiles():
    global saveLog, skyStatusText, testFiles, testNdx, frameStack, solveThisImage
    print("finding files")
    x = datetime.now() + timedelta(seconds=2)
    testFiles = [history_path + '/' + fn for fn in os.listdir(
        history_path) if any(fn.endswith(ext) for ext in ['jpg', 'jpeg', 'png'])]
    testFiles.sort(key=os.path.getmtime)


    print("test files len", len(testFiles))
    frameStack.clear()
    solveLog = [x + '\n' for x in testFiles]
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
    print("restarting thread waiting for 5 seconds", flush=True)
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
    global testMode, testFiles, testNdx, frameStack,  solveLog, state, solveThisImage, skyStatusText

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
def measureTransparency(solveLiveImage = False):
    # tell system to pause taking images after solving the current image
    global skyStatusText, state, solveThisImage, frameStackLock, frameStack,skyStatusText, solveCurrent,\
         solveCompleted,   measureHtmlStack
    solveCurrent = True
    sendStatus('solving')
    if state is Mode.PLAYBACK:
        state = Mode.SOLVETHIS
    
    skyStatusText = "Solving"
    solveCompleted = False
    while not solveCompleted:
        time.sleep(.5)
    skyStatusText = "computing star stats"
    sourcefn = os.path.basename(solveThisImage).split(skyConfig['camera']['format'])[0][:-1]
    try:
        sendStatus('Inspecting stars')
        starlist, image, width, height= Quality.findStarMags(os.path.join(solve_path, imageName),sourcefn)
    except Exception as e:
        print("exception",e)
        traceback.print_exception(*sys.exc_info())
        return 'Image corrupt', ''

    if len(starlist) == 0:
        return starlist,'No named bright stars found'

    #at this point we have found star and thier stats in the image. So now get relative brightness ratios compared to actual ratios.
    try:
        #make magnitude plot curves and return the x,y positions of the stars in the image
        plt.clf()
        results= Quality.plotStarMags(starlist)
    except TypeError as e:
        return(Response('too few stars to plot'))

    #draw color disks indicating how much the flux differes from expected value compared to reference star
    draw = ImageDraw.Draw(image, 'RGBA')

    x = results['ref'][0]
    y = results['ref'][1]
    draw.ellipse((x - 20, y-20,x+20,y+20), outline="yellow")
    for s in results['stars']:
        x = s[2][0]
        y = s[2][1]

        if (s[2][2]< 1.25 and s[2][2] >= 1)  or (s[2][2] < 1 and s[2][2] > .75):
            continue
        if s[2][2] > 1:
            delet = s[2][2]
            color = (10,255,0,40)
        else:
            color = (255,0,0,40)
            delet = 1/s[2][2]

        rad = 6 + 2 * (delet)
        draw.ellipse((x-rad,y-rad,x+rad,y+rad), fill = color)

    if starlist is None:
        skyStatusText = 'No bright stars found'
        return Response(skyStatusText)

    print("removing tmp files")
    for filename in glob.glob(os.path.join(solve_path,'quality*.jpeg')):
        os.remove(filename)
    qfilename = 'quality'+datetime.now().strftime("%m_%d_%y_%H_%M_%S.jpeg")  

    imfn = os.path.join(solve_path, qfilename)
    image.save(imfn)
    print("image saved",flush=True)

    if len(starlist) == 0:
        out = 'no named bright stars found'
    else:

        stars = starlist
        sendStatus('Sorting by Mag')
        # stars.sort(key=lambda x: x['mag'])

        bg = np.asarray([ back['Background'] for back in stars])
        # min = bg.min()
        # max = bg.max()
        mean = bg.mean()
        # std = bg.std()

        # out = '<div style="float:left"><table border = "2"><caption>' + sourcefn + \
        #     ' min:%d Mean:%4.2lf Max:%d std:%4.2lf %dx%d'%(min,mean,max,std,width,height) + \
        #     '</caption><th>Star</th><th>Magnitude</th><th>Flux</th><th>Background</th>'
      
        sendStatus('Making html')

        # for star in stars:

        #     out = out + '<tr><td style="text-align:center">%s </td><td style="text-align:center">%4.2lf</td>'\
        #         '<td style="text-align:center"> %4.2lf</td><td style="text-align:center">%4.2lf</td></tr>' % (
        #             star['name'], star['mag'], star['flux'], star['Background'])


        # out = out + '</table> </div><div " height="Auto">'



        filename = 'qualityplotRatio'+datetime.now().strftime("%m_%d_%y_%H_%M_%S.jpeg") 
        plt.savefig(os.path.join(solve_path, filename ),facecolor='#500000')
        out = '<img src="%s" width="70%%">'%('./static/'+filename)
        out += '<img src="./static/%s" width="70%%"></div>' % (qfilename)

        #skyStatusText = out
    #skyStatusText = out 
    return starlist,out  

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
    
@app.route('/skyQstatus', methods=['POST','GET'])
def skyQStatus():
    global readytoSend,message,allDone
    print("skyall called")
    sendStatus("Ready")

    def inner():
        global  allDone,readytoSend,message
        while True:
            #measureStackLock.acquire()
            if readytoSend:
                yield 'data: '+message+ '\n\n'
                readytoSend = False

                
    return Response(inner(), mimetype='text/event-stream')

    
@app.route('/showImage')
def processImageX():
    name = request.args.get('fn')
    print("processImageX",name)

    html = '<!DOCTYPE html><html><body  style=" background-color: rgb(0, 0, 0)" ><div style="color:#000000" ><p style="color:#000000">'\
    '<img style = "filter:brightness(500%%); color:#000000" src=" ./static/history/%s">'\
    '</p></div></body></html>'%(name)

    return(html)



@app.route("/skyQualitySample", methods=['POST','GET'])
def takeSkyQualitySample():
    global skyStatusText, state, solveThisImage, frameStackLock, frameStack,skyStatusText, solveCurrent,\
         solveCompleted, starMeasures

    # if all parameter is true then use all images in history and rebuild transaprency database
    if request.args.get('live'):
        #turn on solving mode and wait for an image to be solved
        state = Mode.SOLVING
        #return Response("Analizying live image")
    else:
        # stop live solving if running and get history
        if state != Mode.PLAYBACK:
            state = Mode.PLAYBACK
            findHistoryFiles()

    #display the transparancy from the current image
    starlist, out = measureTransparency()

    if len(starlist) == 0:
        skyStatusText = out
        return Response('<body style="background-color: rgb(70, 48, 48); color: lightpink">' + out+'</body>')


    html = out
    #html += '</body></html>'
    skyStatusText = 'Stats complete'
    return Response(html)

def returnImage():
    global testNdx, solveThisImage
    setupImageFromFile()
    filename = 'quality'+datetime.now().strftime("%m_%d_%y_%H_%M_%S.jpeg") 
    copyfile(solveThisImage, os.path.join(solve_path, filename))
    out = '<img style = "filter:brightness(500%%)" src="%s" width="50%%">'%('./static/'+filename)
    print("file name",out,flush=True)
    from PIL import Image, ExifTags

    img = Image.open('./static/' + filename)
    img_exif = img._getexif()

    for key, val in img_exif.items():
        if key in ExifTags.TAGS:
            if 'ExposureTime' in ExifTags.TAGS[key]:
                print('EXP  ',f'{ExifTags.TAGS[key]}:{val[0]/val[1]}')
                break

    sendStatus(out)
    return Response('%d %s'%(testNdx, solveThisImage))

@app.route('/sqDelete', methods=['POST'])
def deletecurrentHistoryImage():
    global testNdx, testFiles
    fn = testFiles[testNdx]
    testFiles.remove(fn)
    os.remove(fn)
    if testNdx > len(testFiles):
        testNdx -= 1
    
    return returnImage()

@app.route('/sqPrevious', methods=['POST'])
def sqPrevious():
    global testNdx
    testNdx -= 1
    if testNdx < 0:
        testNdx = len(testFiles)-1
    return returnImage()

@app.route('/sqNext', methods=['POST'])
def sqNext():
    global testNdx
    testNdx += 1
    if testNdx >= len(testFiles):
        testNdx = 0
    return returnImage()

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
