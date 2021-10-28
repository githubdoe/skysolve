#!/usr/bin/env python3
#
# Based on PSWAI but modified to work with Raspberry PI and immatate SERVO CAT telescope 
# PSWAI: A Plate Solved Where Am I application.
# Copyright (C) 2018 arhcf (user arhcf at github 2018)
# This file is part of the pswai project.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# mimicks a SERVO CAT telescope controler for skysafari.
# will return RA and dec in degrees if given a 'Q' command.
# returns version if given a 'v'
# takes a sky pic and solves and updates ra and dec to solved coords if given a 'g' (goto command)
# Telescope is hard coded to 5005.  IP address is the default address of the RPI on WLAN0 if only one net device.
# you may need to modify the radecFile path to where your plate solve file is.
# you may need to modify the command path to the platesolve script 

# Usage:
# encoder.py [track]
#    track: Causes RA to be modified with time from last plate solve.  Used for scopes that do not track
#           if not set then RA will be a the last value of the plate solve with no adjustment.

 
import sys
import os
import socket
import time
import subprocess
from threading import *

port = 5005     #port for skysafari

radecFile = './static/radec.txt'
track = 0   #default is not to modify RA with time from last solve

# radec.txt file has one line of ra in deg and dec in dec
# lx200 protocol expects the ra to be in hours in decimal notation. 

# reads ra,dec from file and converts it into floatpoint HA and Deg
def radec():
    global track
    ra = 2
    dec = -19
    if not os.access(radecFile, os.R_OK):
      return (ra,dec)
    try:
        fileo=open(radecFile, 'r')
    except:
        print ('no radec file found')
        return (0,0)

    ln = fileo.readline().split()
    print ("line read", ln)
    if (len(ln) < 3 ):
       return (0,0)
    print ('radec is', ln, fileo.name)
    #if ln[3] !='rotation':
        #return (10,10)

    try:
       mytime=ln[0]
       myra=float(ln[1])
       mydec=float(ln[2])
    except (ValueError, IndexError):
       print ("Value error, continuing ....", ln)
       return (0,0)

    myra = myra / 15.  #convert ra in deg back to hours

    lta = time.strftime("%H:%M:%S").split(':')
    tha = (float(lta[0])+float(lta[1])/60+float(lta[2])/3600)
    if track:
        myra = myra - tha

    if (myra >180):
        myra -= 360

    fileo.close()
    print (myra,mydec)
    return myra,mydec

   

radec()
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


argsnum = len(sys.argv)
argi = 1
while (argi < argsnum) :

  if ( sys.argv[argi] == "track") :
     print ("track enabled")
     track = 1
  else :
      print ("Bad option: %s" % (sys.argv[argi]))
      print ("Usage: encoder.py [track] [indi]")
      sys.exit(1)
  argi = argi +1


serversocket.bind(('', port))
#snapthread = Thread(target=snap)
serversocket.listen(5)
print ('server started and listening')
gotoactive = False
while True:
    clientsocket, address = serversocket.accept()
    rbuf=clientsocket.recv(20).decode();
    if rbuf:
       print ("recx:",rbuf,":",hex(ord(rbuf[0])))
       if rbuf[0] == ':':
            print ("recx:",rbuf,":",hex(ord(rbuf[0])))
            if rbuf[1:4] == 'GR#':
                print ('gr#')
                clientsocket.send(b'00:22:00#')
            elif rbuf[1:4] == 'RM#':
                print ('rm#')
                clientsocket.send(b'00:22:00#')
            elif rbuf[1:4] == 'GD#':
                clientsocket.send(b'00:22:00#')

       elif rbuf[0] == 'v':
            print ("recx:",rbuf,":",hex(ord(rbuf[0])))
            clientsocket.send(b"70.A\0");
       elif(rbuf[0] == 'Q' or ord(rbuf[0]) == 0x0d) :  # send ra/dec
            ra,dec = radec()
            out = (b" %06.3lf  %06.3f\0"%(ra,dec))
            print ('xxx',out)
            clientsocket.send(out)
       elif(rbuf[0] == 'H') :  # send encoder resolution
            sbuf=b"8192-8192\r"
            clientsocket.send(sbuf)
       elif(rbuf[0] == 'g'):
            ra,dec = radec()
            ra = 316.
            out = (b"%06.3lf %06.3f"%(ra,dec))
            r = float(rbuf[1:7])
            d = float(rbuf[8:15])
            print (r,d)
            print (rbuf)
            clientsocket.send(out)


       else: print ("recx:",rbuf,":",hex(ord(rbuf[0])))

serversocket.close()
