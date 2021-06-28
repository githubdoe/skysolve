from flask import Flask, render_template, request, Response, send_file
import time

from numpy.core import function_base
from camera_pi import  skyCamera


from fractions import Fraction
import pprint
import configparser
from datetime import datetime
import threading
import numpy
from PIL import Image
import subprocess
import os
import math
import json
from shutil import copyfile
from enum import Enum, auto
import imghdr
import getpass

print ('user', getpass.getuser())
class Mode(Enum):
    PAUSED = auto()
    ALIGN = auto()
    SOLVING = auto()
    PLAYBACK = auto()
    SOLVETHIS = auto()
    AUTOPLAYBACK = auto()   #playback and sovlve history images without user clicking on each.

app = 30
solving = False
maxTime = 50
searchEnable = False
searchRadius = 90
solveLog = []
ra = 0
dec = 0
solveStatus = ''
computedPPa = ''
frameStack = []
frameStackLock = threading.Lock()

focusStd = ''
state = Mode.ALIGN

testNdx = 0
testFiles = []
nextImage=False
root_path = os.getcwd()


solve_path = os.path.join(root_path,'static')
history_path = os.path.join(solve_path, 'history')
demo_path = os.path.join(solve_path, 'demo')
test_path = '/home/pi/pyPlateSolve/data'
history_path = test_path

print ('ff',solve_path)

solveCurrent = False
triggerSolutionDisplay = False
saveObs = False

obsList = []
ndx = 0
lastObs = ""

""" 
skyConfig = {'camera': {'shutter': 1, 'ISO':800, 'Frame': '2000x1500','format': 'jpeg'},
    'solver':{'maxTime': 20, 'solveSigma':9, 'solveDepth':20, 'UselastDelta':False, 'FieldWidthMode':'aPP',
        'FieldWidthModeaPP': 'checked', 'FieldWidthModeField':"", 'FieldWidthModeOther':'', 'aPPValue': 27, 'fieldValue': 14, 'searchRadius': 10,
        'solveVerbose':False},
    'observing':{'saveImages': False, 'showSolution': False, 'savePosition': True , 'obsDelta': .1}}
"""
def saveConfig():
    with open('skyConfig.txt', 'w') as f:
        json.dump(skyConfig, f) 

#saveConfig()
print ('cwd',os.getcwd())
for i in range (130):
    solveLog.append("                       \n")

with open(os.path.join(root_path,'skyConfig.txt')) as f:
    skyConfig = json.load(f)



imageName = 'cap.'+skyConfig['camera']['format']

#print (skyConfig)
skyCam = None

def solveThread():
    global skyStatusText, focusStd, solveCurrent,state, skyCam ,frameStack, frameStackLock, testNdx
    print('solvethread')
    while True:
        if state is Mode.PAUSED or state is Mode.PLAYBACK:
            continue
        if state is Mode.SOLVETHIS:
            if len(frameStack) == 0:
                continue
            frame = frameStack[-1][0]
            frameStack.clear()
            with open(os.path.join(solve_path,imageName), "wb") as f:
                try:
                    f.write(frame)
                except Exception as e:
                    print (e)
                    solveLog.ap
                if not solve(os.path.join(solve_path,imageName)):
                    #try again but this time since the previous failed it will not use a starting guess possition
                    solve(os.path.join(solve_path,imageName))   
                state = Mode.PLAYBACK
                continue
                    
        else:
            frame = skyCam.get_frame()


        if (state is Mode.AUTOPLAYBACK):
            if testNdx == len(testFiles):
                testNdx = 0
            fn = testFiles[testNdx]
            testNdx += 1

            print ("image", testNdx, fn)


            with open(fn, 'rb') as infile:
                frame = infile.read()

        with open(os.path.join(solve_path,imageName), "wb") as f:
            try:
                f.write(frame)
            except Exception as e:
                print (e)
                solveLog.append(str(e) + '\n')
                continue

        frameStackLock.acquire()

        frameStack.append((frame,datetime.now())) 
        frameStackLock.release()

        if state is Mode.SOLVING or state is Mode.AUTOPLAYBACK:
            solve(os.path.join(solve_path,imageName))
            continue    
        #else measaure contrast for focus bar



        try:
            img = Image.open(os.path.join(solve_path,imageName)).convert("L")
            imgarr = numpy.array(img) 
            avg = imgarr.mean()

            focusStd = f"{100 * imgarr.std()/avg:.2f}"

        except Exception as e:
            print (e)

def solve(fn, parms = []):
    global app, solving, maxTime, searchRaius, solveLog, ra, dec, searchEnable, solveStatus,\
        triggerSolutionDisplay, skyStatusText, lastObs
    startTime = datetime.now()
    solving = True
    solved = ''
    fieldwidthParm = ''
    if skyConfig['solver']['FieldWidthMode'] == 'FieldWidthModeApp':
        field = ['-u', 'app', '-L', str(skyConfig['solver']['appValue'])]
    elif skyConfig['solver']['FieldWidthMode'] == 'FieldWidthModeField':
        field = ['u', 'degw', '-L', str(skyConfig['solver']['fieldValue'])]    
    else: field = []
    parms = parms + field
    parms = parms + ['--cpulimit', str(skyConfig['solver']['maxTime'])]
    if skyConfig['solver']['searchRadius']>0  and ra != 0:
        parms = parms + ['--ra', str(ra), '--dec', str(dec), '--radius', str(skyConfig['solver']['searchRadius'])]
    print('show stars', skyConfig['solver']['showStars'])
    if not skyConfig['solver']['showStars']:
        parms = parms + ['-p']
    cmd = ["solve-field", fn,"--depth" ,str(skyConfig['solver']['solveDepth']), "--sigma", str(skyConfig['solver']['solveSigma']),
        '--overwrite'] + parms

    print ("solving ",cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    solveLog.clear()
    ppa = ''
    starNames={}
    duration = None
    radec = ''
    ra = 0
    dec = 0
    solveLog.append("solving:\n")
    while not p.poll():
        stdoutdata = p.stdout.readline().decode(encoding='UTF-8')
        if stdoutdata:
            if skyConfig['solver']['solveVerbose']:
                solveLog.append(stdoutdata)
            #print ("stdout", str(stdoutdata))

            if stdoutdata.startswith('Field center: (RA,Dec) = ('):
                solved = stdoutdata
                fields = solved.split()[-3:-1]
                print ('f',fields)
                ra = fields[0][1:-1]
                dec = fields[1][0:-1]
                ra = float(fields[0][1:-1])
                dec = float(fields[1][0:-1])
                radec="%s %6.6lf %6.6lf \n"%(time.strftime('%H:%M:%S'),ra,dec)
                file1 = open(os.path.join(solve_path,"radec.txt"),"w")#write mode 
                file1.write(radec) 
                file1.close() 
                stopTime = datetime.now()
                duration = stopTime - startTime
                #print ('duration', duration)

            elif stdoutdata.startswith('Field size'):
                print ("Field size")
                solveLog.append(stdoutdata)
                solveStatus +=(". " + stdoutdata.split(":")[1].rstrip())
            elif stdoutdata.find('pixel scale') > 0:
                computedPPa = stdoutdata.split("scale ")[1].rstrip()
            elif 'The star' in stdoutdata:
                stdoutdata = stdoutdata.replace(')','')
                con = stdoutdata[-4:-1]
                if con not in starNames:
                    starNames[con]=1

        else:
            break

    solveStatus+= ". scale " + ppa
    
    if not solved:
        skyStatusText = "Failed"
        ra = 0

    else:
        # Write-Overwrites 
        file1 = open(os.path.join(solve_path,"radec.txt"),"w")#write mode 
        file1.write(radec) 
        file1.close() 
        cmdPlot = ['/usr/bin/plot-constellations', '-v' ,'-w' ,os.path.join(solve_path,'cap.wcs'),\
         '-B', '-C', '-N', '-o' , os.path.join(solve_path,'cap-ngc.png')]
        p2 = subprocess.Popen(cmdPlot ,stdout=subprocess.PIPE)
        ndx=0
        stars = []
        while not p2.poll():
            if ndx > 1000:
                break
            ndx += 1
            stdoutdata = p2.stdout.readline().decode(encoding='UTF-8')
            if stdoutdata:
                stars.append(stdoutdata)
                #print (stdoutdata)
                if 'The star' in stdoutdata:
                    stdoutdata = stdoutdata.replace(')','')
                    con = stdoutdata[-4:-1]
                    if con not in starNames:
                        starNames[con]=1
        foundStars = ', '.join(stars).replace("\n","")
        foundStars = foundStars.replace("The star","")
        solveLog.append(foundStars + "\n")
        constellations = ', '.join(starNames.keys())

        if skyConfig['observing']['savePosition']:
            obsfile = open(os.path.join(solve_path,"obs.log"),"a+")
            obsfile.write(radec.rstrip() + " " + constellations + '\n')
            obsfile.close()
            print ("wrote obs log")

        if skyConfig['observing']['saveImages']:
            saveimage = False
            if skyConfig['observing']['obsDelta'] > 0:
                print ("checking delta")
                if lastObs:
                    old = lastObs.split(" ")
                    ra1 = math.radians(float(old[1]))
                    dec1 = math.radians(float(old[2]))
                    ra2 = math.radians(float(ra))
                    dec2 = math.radians(float(dec))

                    delta = math.degrees(math.acos( math.sin(dec1) * math.sin(dec2) + math.cos(dec1)* math.cos(dec2)*math.cos(ra1 - ra2)))
                    if delta > skyConfig['observing']['obsDelta']:
                        saveimage = True
                else:
                    saveimage = True
            if saveimage:
                fn = datetime.now().strftime("%m_%d_%y_%H_%M_%S.")+skyConfig['camera']['format']
                copyfile(os.path.join(solve_path,imageName), os.path.join(solve_path, 'history',fn))


        lastObs = radec
        if skyConfig['observing']['showSolution']:
            triggerSolutionDisplay = True
        skyStatusText = foundStars + " "+str(duration) +' secs'


    solving = False

    return solved    

app = Flask(__name__)

skyStatusText = 'Initilizing Camera'

@app.route("/" , methods=['GET','POST' ])
def index():
    global skyCam
    shutterValues = ['.01', '.05', '.1','.15','.2','.5','.7','.9','1', '2.', '3','4','5','10']
    skyFrameValues = ['400x300', '640x480', '800x600', '1024x768', '1280x960', '1920x1440', '2000x1000', '2000x1500']
    isoValues = ['100','200','400','800']
    formatValues=['jpeg','png']
    solveParams = { 'PPA': 27, 'FieldWidth': 14, 'Timeout': 34, 'Sigma': 9 , 'Depth':20, 'SearchRadius': 10}
    if not skyCam:
        print('creating cam')
        skyCam= skyCamera(shutter  = int(1000000 * float(skyConfig['camera']['shutter'])), format=skyConfig['camera']['format'] ,resolution = skyConfig['camera']['frame'])
    
    return render_template('template.html', shutterData = shutterValues, skyFrameData = skyFrameValues, skyFormatData = formatValues,
            skyIsoData = isoValues, solveP = skyConfig['solver'], obsParms = skyConfig['observing'], cameraParms=skyConfig['camera'])

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

@app.route('/pause', methods = ['POST'])
def pause():
    global skyStatusText, skyCam, state
    if skyCam:
        skyCam.pause()
    state = Mode.PAUSED

    skyStatusText = 'Paused'
    return Response(skyStatusText)

@app.route('/Align', methods = ['POST'])
def Align():
    global skyStatusText, skyCam, state
    if skyCam:
        skyCam.resume()
    if state is Mode.PLAYBACK:
        skyStatusText = 'you must [press replay images first to Cancel playback].'
    else:
        state = Mode.ALIGN
        skyStatusText = 'Align Mode'
    return Response(skyStatusText)

@app.route('/Solve', methods = ['POST'])
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



@app.route('/setISO/<value>', methods = ['POST'])
def setISO(value):
    global solveLog, skyStatusText, isoglobal
    solveLog.append("ISO changing will take 10 seconds to stabilize gain.\n")
    skyStatusText = "changing to ISO " + value
    isoglobal = value
    skyCam.setISO(value)

    skyConfig['camera']['ISO'] = value
    saveConfig()

    return Response(status = 204)

@app.route('/setFrame/<value>', methods = ['POST'])
def setFrame(value):
    skyCam.setResolution(value)
    skyConfig['camera']['frame'] = value
    saveConfig()
    return Response(status = 204)

@app.route('/setFormat/<value>', methods = ['POST'])
def setFormat(value):
    global skyStatusText, imageName
    skyStatusText = "changing Image Format to " + value
    imageName = 'cap.'+value
    skyCam.setFormat(value)
    skyConfig['camera']['format'] = value
    saveConfig()

    return Response(status = 204)

@app.route('/setShutter/<value>', methods = ['POST'])
def setShutter(value):
    print ("shutter value", value)
    skyCam.setShutter(int(1000000 * float(value)))
    skyConfig['camera']['shutter'] = value
    saveConfig()
    return Response(status = 204)

@app.route('/showSolution/<value>',methods = ['POSt'])
def setShowSolutions(value):
    global skyConfig
    print ('show solutions', value)
    if (value == '1'):
        skyConfig['observing']['showSolution']= value == '1'
    saveConfig()
    return Response(status = 204)


@app.route('/saveImages/<value>',methods = ['POST'])
def saveSolvedImage(value):
    global skyConfig
    print ('save images ', value)
    skyConfig['observing']['saveImages'] = value == '1'
    saveConfig()
    print ("config", skyConfig)
    return Response(status = 204)
@app.route('/clearObsLog', methods = ['POST'])

def clearObsLog():
    global ndx, obsList
    ndx = 0
    obsList.clear()
    with open(os.path.join(solve_path,'obs.log'), 'w') as infile:
        pass #this will write an empty file erasing the previous contents
    return Response("observing log cleared")


    
@app.route('/skyStatus', methods=['post'])
def skyStatus():
    global skyStatusText
    return Response(skyStatusText)

@app.route('/Focus', methods=['post'])
def Focus():
    global focusStd
    return Response(focusStd)

solveT = None
def gen():
    global skyStatusText, solveT, testNdx,nextImage, triggerSolutionDisplay, testMode, state, solveCurrent, frameStackLock
    #Video streaming generator function.

    if not solveT:  #start up solver if not already running.
        solveT = threading.Thread( target = solveThread)
        solveT.start()
    lastImageTime = datetime.now()
    while True:
        if state is Mode.ALIGN or state is Mode.SOLVING:
            while len(frameStack) == 0:
                if state is Mode.PLAYBACK or state is Mode.SOLVETHIS:
                    break
            if state is Mode.PLAYBACK or state is Mode.SOLVETHIS:
                continue

            frameStackLock.acquire()
            if frameStack[-1][1] == lastImageTime or len(frameStack) == 0:
                frameStackLock.release()
                continue

            lastImageTime = frameStack[-1][1]
            frame = frameStack[-1][0]
            frameStack.clear()
            frameStackLock.release()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if state is Mode.SOLVING:
                solveCurrent = True
        else:

            if nextImage:
                nextImage = False
                if testNdx == len(testFiles):
                    testNdx = 0
                fn = testFiles[testNdx]

                #print ("image", testNdx, fn)
                skyStatusText = "%d %s"%(testNdx,fn)

                with open(fn, 'rb') as infile:
                    frame = infile.read()
                    frameStack.append((frame, datetime.now()))
                    solveLog.append('next image ' + fn + '\n')
                    yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


            elif triggerSolutionDisplay and skyConfig['observing']['showSolution']:
                triggerSolutionDisplay = False
                fn = os.path.join(solve_path, 'cap-ngc.png')

                #skyStatusText = " solution %s"%(fn)
                with open(fn, 'rb') as infile:
                    frame = infile.read()


                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(2)
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        #print ('no action', state, nextImage)

@app.route('/applySolve2', methods=['POST'])
def apply():
    print (request.form.values)
    req = request.form

    mode = skyConfig['solver']['FieldWidthMode']= req.get("FieldWidthMode")
    skyConfig['solver']['FieldWidthModeaPP'] = skyConfig['solver']['FieldWidthModeField'] = skyConfig['solver']['FieldWidthModeOther'] = ''
    skyConfig['solver'][mode] = 'checked'
    print ('config', mode, skyConfig)
    skyConfig['solver']['fieldValue'] = float(req.get("fieldValue"))
    skyConfig['solver']['aPPValue'] = float(req.get("aPPValue"))
    skyConfig['solver']['searchRadius'] = float(req.get("searchRadius"))
    skyConfig['solver']['solveSigma'] = int(req.get("solveSigma"))
    skyConfig['solver']['solveDepth'] = int(req.get("solveDepth"))
    skyConfig['solver']['solveVerbose'] = bool(req.get("solveVerbose"))
    skyConfig['solver']['showStars'] = bool(req.get("showStars"))
    saveConfig()

    #print (request.form.['submit_button'])
    return Response(status=204)

@app.route('/demoMode', methods=['POST'])
def demoMode():
    global testMode, testFiles, testNdx, nextImage, frameStack,  solveLog, state

    skyStatusText = 'Demo images will be set for playback'
    state = Mode.PLAYBACK
    testFiles = [demo_path + '/' + fn for fn in os.listdir(demo_path)  if any(fn.endswith(ext) for ext in ['jpg', 'png'])]
    testFiles.sort(key=os.path.getmtime)
    testNdx = 0
    nextImage = True
    print("demo files len", len(testFiles))
    frameStack.clear()
    solveLog = [  x+ '\n' for x in testFiles]

    return Response(skyStatusText)

@app.route('/testMode', methods=['POST'])
def toggletestMode():
    global testMode, testFiles, testNdx, nextImage, frameStack,  solveLog, state

    if  not state is  state.PLAYBACK:
        skyStatusText = 'PLAYBACK mode'
        state = Mode.PLAYBACK
        testFiles = [history_path + '/' + fn for fn in os.listdir(history_path)  if any(fn.endswith(ext) for ext in ['jpg','jpeg', 'png'])]
        testFiles.sort(key=os.path.getmtime)
        testNdx = 0
        nextImage = True
        print("test files len", len(testFiles))
        frameStack.clear()
        solveLog = [  x+ '\n' for x in testFiles]
        if len(testFiles) == 0:
            skyStatusText("no image files in hisotry")
        else :
            skyStatusText = 'PLAYBACK mode ' + str(len(testFiles)) + " images found."
        
    else:
        state = Mode.ALIGN
        skyStatusText = 'ALIGN Mode'
    return Response(skyStatusText)

@app.route('/nextImage', methods=['POST'])
def nextImagex():
    global nextImage, testNdx
    nextImage = True
    testNdx += 1
    if testNdx >= len(testFiles):
        testNdx = 0

    skyStatusText="next image is "+ str(testNdx)
    return Response(skyStatusText)

@app.route('/retryImage', methods=['POST'])
def retryImage():
    global nextImage, testNdx
    nextImage = True
    skyStatusText="retry image is "+ str(testNdx)
    time.sleep(3)
    return Response(skyStatusText)

@app.route('/prevImage', methods=['POST'])
def prevImage():
    global nextImage, testNdx
    if (testNdx > 0):
        testNdx -= 1
    else:
        testNdx = len(testFiles)-1

    nextImage = True
    skyStatusText="next image is "+ str(testNdx)
    return Response(skyStatusText)

@app.route('/Observe', methods=['POST'])
def setObserveParams():
    print (request.form.values, request.form.getlist('saveImages'))
    skyStatusText = "observing session parameters received."
    return Response(status=205)

@app.route('/startObs', methods=['POST'])
def startObs():
    global ndx, obsList
    ndx = 0
    with open(os.path.join(solve_path,'obs.log'), 'r') as infile:
        obsList = infile.readlines()

    return updateRADEC()

def updateRADEC():
    global ndx, obsList
    radec = obsList[ndx]
    file1 = open(os.path.join(solve_path,"radec.txt"),"w")#write mode 
    file1.write(radec) 
    file1.close() 
    return Response(radec.rstrip())

@app.route('/nextObs', methods=['POST'])
def nextObs():
    global ndx, obsList
    ndx +=1
    if ndx > len(obsList):
        ndx-=1
    return updateRADEC()

@app.route('/prevObs', methods=['POST'])
def prevObs():
    global ndx, obsList
    ndx -=1
    if ndx < 0:
        ndx = 0
    return updateRADEC()

@app.route('/solveThis', methods=['POSt'])
def solveThis():
    global solveCurrent, state
    solveCurrent = True
    state = Mode.SOLVETHIS

    skyStatusText="Solving"
    return Response(skyStatusText)
from zipfile import ZipFile


# Zip the files from given directory that matches the filter
def zipFilesInDir(dirName, zipFileName):
   # create a ZipFile object
   with ZipFile(zipFileName, 'w') as zipObj:
        valid_images = [".jpg",".png",".jpeg"]
        imgs = []
        for f in os.listdir(dirName):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            imgs.append(f)
        print ("images", imgs)
        for filename in imgs:
            # create complete filepath of file in directory
            filePath = os.path.join(solve_path,'history', filename)
            # Add file to zip
            zipObj.write(filePath,filename)

@app.route('/clearImages', methods=['POST'])
def clearImages():
    global solve_path
    valid_images = [".jpg",".png", ".jpeg"]
    imgs = []
    for f in os.listdir(os.path.join(solve_path, 'history')):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        os.remove(os.path.join(solve_path, 'history',f));
    return Response("images deleted")

@app.route('/zip', methods=['GET'])
def zipImages():
    global solve_path
    print ("download zip file")
    zipFilesInDir(os.path.join(solve_path, 'history'), os.path.join(solve_path,'history','history.zip'))
    return send_file(os.path.join(solve_path,'history','history.zip'),
                     attachment_filename='history.zip',
                     mimetype='application/zip')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("working dir", os.getcwd())

    app.run(host='0.0.0.0', debug=True)
    
