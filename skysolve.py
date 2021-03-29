from flask import Flask, render_template, request, Response
import time
from camera_pi import Camera, FileCamera
import picamera
from fractions import Fraction
import pprint
import base_camera
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

app = 30
solving = False
maxTime = 50
searchEnable = False
searchRadius = 90
solveLog = []
ra = 0
dec = 0
lastRA = ''
solveStatus = ''
computedPPa = ''
frameStack = []
nameStack = []
focusStd = ''
testMode=False
testNdx = 0
testFiles = []
nextImage=False
root_path = os.getcwd()
print ('roor',root_path)
test_path = '/home/pi/pyPlateSolve/data'
solve_path = os.path.join(root_path,'static')
print ('ff',solve_path)

solveCurrent = False
showSolution = False
triggerSolutionDisplay = False
saveObs = False
flag = False
obsList = []
ndx = 0

skyConfig = {'camera': {'shutter': 1},
    'solver':{'maxTime': 20, 'sigma':9, 'depth':20, 'UselastDelta':False},
    'observing':{'saveImage': True, 'savePosition': True} }


def solveThread():
    global skyStatusText, focusStd, solveCurrent, flag
    print ('solveThread()')
    while True:
        if flag:
            print ("framestack", len(frameStack))
            flag = False
        while len(frameStack) == 0:  
            if solveCurrent:
                solveCurrent = False
                print ('solveing source')
                solve(os.path.join(solve_path,'cap.jpg'))
                continue    
            pass
        #skyStatusText = datetime.now().strftime("%H:%M:%S")
        #skyStatusText = FileCamera.current

        print ('solveThread image', skyStatusText, len(frameStack))
        #with open("cap.jpg", "wb") as f:
            #f.write(frame)
        if len(frameStack) == 0:
            print ('empty stack', len(frameStack))
            continue
        with open(os.path.join(solve_path,"cap.jpg"), "wb") as f:
            try:
                f.write(frameStack.pop())
                name = nameStack.pop()
            except Exception as e:
                print (e)
                solveLog.append(str(e) + '\n')
                continue
            solveLog.append(name + '\n')
            print ("wrote file",f.name,name)
            print ('solve thread')


        if len(frameStack)> 0:
            print ('stack  not empty', len(frameStack))
            frameStack.clear()
            nameStack.clear()

        try:
            img = Image.open("cap.jpg").convert("L")
            imgarr = numpy.array(img) 

            focusStd = f"{imgarr.std():.2f}"

        except Exception as e:
            print (e)

def solve(fn, parms = []):
    global app, solving, maxTime, searchRaius, solveLog, ra, dec, searchEnable, lastRA, solveStatus,\
        showSolution, triggerSolutionDisplay, skyStatusText
    solving = True
    solved = ''

    parms = parms + ['-u', 'app', '-L', str(app), '--cpulimit', str(maxTime)]
    if searchEnable and lastRA != '':
        parms = parms + ['--ra', ra, '--dec', dec, '--radius', str(searchRadius)]
    cmd = ["solve-field", fn,"--depth" ,"20", "--sigma", "9", '--overwrite'] + parms
    print ("solving ",cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    solveLog.clear()
    ppa = ''
    starNames={}
    radec = ''
    while not p.poll():
        stdoutdata = p.stdout.readline().decode(encoding='UTF-8')
        if stdoutdata:
            solveLog.append(stdoutdata)
            print ("stdout", str(stdoutdata))

            if stdoutdata.startswith('Field center: (RA,Dec) = ('):

                solved = stdoutdata
                fields = solved.split()[-3:-1]
                print ('f',fields)
                ra = fields[0][1:-1]
                dec = fields[1][0:-1]
                ra = float(fields[0][1:-1])
                dec = float(fields[1][0:-1])

                radec="%s %6.6lf %6.6lf \n"%(time.strftime('%H:%M:%S'),ra,dec)



            elif stdoutdata.startswith('Field size'):
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
    
    print("solution results", solved, showSolution)
    if not solved:
        skyStatusText = "Failed"

    else:
        # Write-Overwrites 
        file1 = open(os.path.join(solve_path,"radec.txt"),"w")#write mode 
        file1.write(radec) 
        file1.close() 
        constellations = ', '.join(starNames.keys())
        print (' config was',skyConfig['observing']['savePosition'])
        if skyConfig['observing']['savePosition']:
            obsfile = open(os.path.join(solve_path,"obs.log"),"a+")
            obsfile.write(radec.rstrip() + " " + constellations + '\n')
            obsfile.close()
            print ("wrote obs log")

        if skyConfig['observing']['saveImage']:
            fn = datetime.now().strftime("%m_%d_%y_%H:%M:%S")+'.jpg'
            copyfile(os.path.join(solve_path,"cap.jpg"), os.path.join(solve_path, 'history',fn))


        if showSolution:
            triggerSolutionDisplay = True
        skyStatusText = radec.rstrip() + " " + constellations
        print ("from solve skystatusText is",skyStatusText)
        
    solving = False

    return solved    

app = Flask(__name__)

skyStatusText = 'Initilizing Camera'

@app.route("/" , methods=['GET','POST' ])
def index():
    shutterValues = ['.01', '.05', '.1','.15','.2','.5','.7','1','1.5', '2.', '3','4','5','10']
    skyFrameValues = ['400x300', '640x480', '800x600', '1024x768', '1280x720', '1920x1080', '2000x1000']
    isoValues = ['100','200','400','800']
    solveParams = { 'PPA': 27, 'FieldWidth': 14, 'Timeout': 34, 'Sigma': 9 , 'Depth':20, 'SearchRadius': 10}

    return render_template('template.html', shutterData = shutterValues, skyFrameData = skyFrameValues,
            skyIsoData = isoValues, solveP = solveParams)

# 

@app.route("/solveLog")
def streamSolveLog():
    global solveLog
    print("solveLog")
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
    global skyStatusText
    print("put")
    skyStatusText = 'Paused'
    return Response(skyStatusText)

@app.route('/Align', methods = ['POST'])
def Align():
    global skyStatusText
    print("put", request)
    skyStatusText = 'Align Mode'
    return Response(skyStatusText)

@app.route('/Solve', methods = ['POST'])
def Solving():
    global skyStatusText
    print("put")
    skyStatusText = 'Solving Mode'
    return Response(skyStatusText)

@app.route('/setShutter/<value>', methods = ['POST'])
def setShutter(value):
    print ("shutter value", value)
    Camera.theCam.shutter_speed = int(1000000 * float(value))
    return Response(status = 204)

@app.route('/showSolution/<value>',methods = ['POSt'])
def setShowSolutions(value):
    global showSolution
    print ('show solutions', value)
    if (value == '1'):
        showSolution = True
    else :
        showSolution = False
    print ("showSolution is", showSolution)
    return Response(status = 204)

@app.route('/setISO/<value>', methods = ['POST'])
def setISO(value):
    print ("ISO value", value)
    Camera.theCam.iso = int(value)
    return Response(status = 204)

def get_message():
    '''this could be any function that blocks until data is ready'''
    time.sleep(1.0)
    s = time.ctime(time.time())
    return s

cnt = 0
@app.route('/skyStatus', methods=['post'])
def skyStatus():
    global skyStatusText
    print ("sky status is:", skyStatusText)
    return Response(skyStatusText)

@app.route('/Focus', methods=['post'])
def Focus():
    global focusStd
    return Response(focusStd)


solveT = None
def gen(camera):
    global skyStatusText, solveT, testNdx,nextImage, triggerSolutionDisplay, flag, testMode
    print ('gen called', camera)
    """Video streaming generator function."""
    
    if not solveT:  #start up solver if not already running.
        solveT = threading.Thread( target = solveThread)
        solveT.start()
 
    while True:
        #print ("gen ",nextImage, testMode)
        if not testMode:
            frame = camera.get_frame()
            skyStatusText = "Camera running"
            frameStack.append(frame)
            nameStack.append('camera')
            flag = True
            #with open("cap.jpg", "wb") as f:
                #f.write(frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if nextImage:
                print ("gen next image")
                nextImage = False
                if testNdx == len(testFiles):
                    testNdx = 0
                fn = testFiles[testNdx]

                print ("image", testNdx, fn)
                skyStatusText = "%d %s"%(testNdx,fn)

                with open(fn, 'rb') as infile:
                    frame = infile.read()
                    frameStack.append(frame)
                    nameStack.append(fn)
                    solveLog.append('next image ' + fn)
                    yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


            elif triggerSolutionDisplay and showSolution:
                print ("show solution")
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

@app.route('/applySolve2', methods=['POST'])
def apply():
    print ("submit")
    print (request.form.values)
    req = request.form
    print ("fw =" ,req.get("FW"), "appValue = ",req.get("aPPValue"))
    #print (request.form.['submit_button'])
    return Response(status=204)

@app.route('/testMode', methods=['POST'])
def toggletestMode():
    global testMode, testFiles, testNdx, nextImage, frameStack, nameStack, solveLog
    testMode =  not testMode
    print("testmodeccc", testMode)
    if (testMode):
        skyStatusText = 'Test mode'
        testFiles = [test_path + '/' + fn for fn in os.listdir(test_path)  if any(fn.endswith(ext) for ext in ['jpg', 'png'])]
        testFiles.sort(key=os.path.getmtime)
        testNdx = 0
        nextImage = True
        print("test files len", len(testFiles))
        frameStack.clear()
        nameStack.clear()
        solveLog = [  x+ '\n' for x in testFiles]

    else:
        skyStatusText = 'exit text mode'
    return Response(skyStatusText)

@app.route('/nextImage', methods=['POST'])
def nextImagex():
    global nextImage, testNdx
    nextImage = True
    testNdx += 1
    print ('next image pressed')
    skyStatusText="next image is "+ str(testNdx)
    return Response(skyStatusText)

@app.route('/retryImage', methods=['POST'])
def retryImage():
    global nextImage, testNdx
    nextImage = True
    print ('retry image pressed')
    skyStatusText="retry image is "+ str(testNdx)
    return Response(skyStatusText)

@app.route('/prevImage', methods=['POST'])
def prevImage():
    global nextImage, testNdx
    if (testNdx > 0):
        testNdx -= 1
    print ('prev ndx', testNdx ,testFiles[testNdx])
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
    global solveCurrent
    solveCurrent = True
    print ("solve current set")
    skyStatusText="Solving"
    return Response(skyStatusText)



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    #return Response(gen(FileCamera()),
    return Response(gen(Camera((2000,1000), 1000000, 800)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("working dir", os.getcwd())

    app.run(host='0.0.0.0', debug=True)
