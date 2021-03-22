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

app = 15
solving = False
maTime = 20
searchEnable = False
searchRadius = 90
solveLog = []
ra = 0
dec = 0
lastRA = ''
saveSolution = ''
solveStatus = ''
computedPPa = ''
frameStack = []
focusStd = ''
maxTime = 25
testMode=False
testNdx = 0
testFiles = []
nextImage=False
test_path = '/home/pi/pyPlateSolve/data'
solve_path = '/home/pi/pyPlateSolve/skySolve/SolvedData'
solveCurrent = False
showSolution = False
triggerSolutionDisplay = False

source_Image = ''



def solveThread():
    global skyStatusText, focusStd, solveCurrent, source_Image
    print ('solveThread()')
    while True:
        while len(frameStack) == 0:  
            if solveCurrent:
                solveCurrent = 0
                print ('solveing source', source_Image)
                solve('cap.jpg')
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
        with open("cap.jpg", "wb") as f:
            source_image = frameStack.pop()
            f.write(source_image)
            print ("wrote file",f.name)
        print ('solve thread')
        if solveCurrent:
            print ('solving current')
            solve('cap.jpg')
            solveCurrent = False

        if len(frameStack)> 0:
            print ('stack  not empty', len(frameStack))
            frameStack.clear()

        try:
            img = Image.open("cap.jpg").convert("L")
            imgarr = numpy.array(img) 

            focusStd = f"{imgarr.std():.2f}"

        except Exception as e:
            print (e)

def solve(fn, parms = []):
    global app, solving, maxTime, searchRaius, solveLog, ra, dec, searchEnable, lastRA, solveStatus,\
        showSolution, triggerSolutionDisplay
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
    while not p.poll():
        stdoutdata = p.stdout.readline().decode(encoding='UTF-8')
        if stdoutdata:
            solveLog.append(stdoutdata)
            print ("stdout", str(stdoutdata))
            if stdoutdata.startswith('Field center: (RA,Dec) = ('):
                saveSolution = stdoutdata
                solved = stdoutdata
            elif stdoutdata.startswith('Field size'):
                solveStatus +=(". " + stdoutdata.split(":")[1].rstrip())
            elif stdoutdata.find('pixel scale') > 0:
                computedPPa = stdoutdata.split("scale ")[1].rstrip()
        else:
            break

    solveStatus+= ". scale " + ppa
    
    skyStatusText = solveStatus
    print("solution results", solved, showSolution)
    if not solved:
        solveStatus.Clear()
        solveStatus.append("Failed")
    elif showSolution:
        triggerSolutionDisplay = True;
        print ("showing solution")
        
    solving = False
    
    return solved    


app = Flask(__name__)

skyStatusText = 'Initilizing Camera'

@app.route("/" , methods=['GET','POST' ])
def index():
    shutterValues = ['.01', '.05', '.1','.15','.2','.5','.7','1','1.5', '2.', '3','4','5','10']
    skyFrameValues = ['400x300', '640x480', '800x600', '1024x768', '1280x720', '1920x1080', '2000x1000']
    isoValues = ['100','200','400','800']

    return render_template('template.html', shutterData = shutterValues, skyFrameData = skyFrameValues,
            skyIsoData = isoValues)
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
                print ("log", str)
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
import math
ccc = 1
@app.route('/log')
def solvelog():
    global ccc
    def generate():
        global ccc
        ccc = ccc + 1
        return ()


    return app.response_class(generate(), mimetype='text/plain')

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
    global skyStatusText, solveT, testNdx,nextImage, triggerSolutionDisplay
    print ('gen called', camera)
    """Video streaming generator function."""
    
    if not solveT:  #start up solver if not already running.
        solveT = threading.Thread( target = solveThread)
        solveT.start()
 
    while True:
        if not testMode:

            frame = camera.get_frame()
            skyStatusText = "Camera running"
            frameStack.append(frame)
            with open("cap.jpg", "wb") as f:
                f.write(frame)


            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if nextImage:

                nextImage = False
                

                if testNdx == len(testFiles):
                    testNdx = 0
                fn = testFiles[testNdx]

                print ("image", testNdx, fn)
                skyStatusText = "%d %s"%(testNdx,fn)

                with open(fn, 'rb') as infile:
                    frame = infile.read()
                    frameStack.append(frame)

                    yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
                    time.sleep(4)

            elif triggerSolutionDisplay and showSolution:
                print ("show solution")
                triggerSolutionDisplay = False
                fn = os.path.join(solve_path, 'cap-ngc.png')

                skyStatusText = " solution %s"%(fn)
                with open(fn, 'rb') as infile:
                    frame = infile.read()


                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(2)
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/applySolve', methods=['POST'])
def apply():
    print ("submit")
    print (request.form.values)
    #print (request.form.['submit_button'])
    return Response(status=204)

@app.route('/testMode', methods=['POST'])
def toggletestMode():
    global testMode, testFiles, testNdx, nextImage
    testMode =  not testMode
    print("testmodeccc", testMode)
    if (testMode):
        skyStatusText = 'Test mode'
        testFiles = [test_path + '/' + fn for fn in os.listdir(test_path)  if any(fn.endswith(ext) for ext in ['jpg', 'png'])]
        testFiles.sort(key=os.path.getmtime)
        testNdx = 0
        nextImage = True
        print("test files len", len(testFiles))
    else:
        skyStatusText = 'exit text mode'
    return Response(skyStatusText)

@app.route('/nextImage', methods=['POST'])
def nextImage():
    global nextImage, testNdx
    nextImage = True
    testNdx += 1
    skyStatusText="next image is "+ str(testNdx)
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
    os.chdir(solve_path)
    app.run(host='0.0.0.0', debug=True)
