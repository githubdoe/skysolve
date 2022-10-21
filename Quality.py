"""
Finds the brightness and magnitude of solved images stored in cap.jpeg alaong with the solved data in cap.wcs

Note: Requires PIL (pip install Pillow)
"""
from cmath import nan
from logging import exception
from operator import truediv
from termios import IXANY
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import os
from pathlib import Path
import subprocess
import tetra3
import sys
import json as json
import pprint
import bstars
import math
import re
import constellations
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, spatial
from tinydb import TinyDB, Query
import fitsio
from fitsio import FITS,FITSHDR

sys.path.append('..')
brightNamesStars = None
kdtree = None
hiptree = None
hiptable = None
# find star magnitude and name of stars in image.  Uses astrometry's corr.cap file 
def findClosestPoint(x1,y1,starx, stary):
    bestdel = 10000
    bestNdx = -1
    for ndx, x in enumerate(starx):
        dx = 20 * (x - x1)
        dy = stary[ndx] - y1
        delta = math.sqrt(dx * dx + dy * dy)
        if delta < bestdel:
            bestdel = delta
            bestNdx = ndx
    return bestNdx
def findStarMags(fn, sourceOfImage= None):
    global brightNamesStars, kdtree

    def getHipStar(ra,dec):
        global hiptree,hiptable
        if hiptree is None:
            hipTable,h= fitsio.read('hip.fits',header=True)
            data = [[r[1],r[0]] for r in hipTable[:10000]]

            hiptree = spatial.KDTree(data[0::200])
        delta,index = hiptree.query([ra,dec])

        return 'Hip%d'%(index)


    def getStarName(ra,dec,tree):
        delta,index = tree.query([ra,dec],1)
        if delta < .1:
            star = brightNamesStars[index]
            if star[1] != b'':
                return star[1]
            elif star[0] == b'':
                return 'B%d'%(index)
            else:
                return star[0].decode('utf-8')


    if not brightNamesStars:
        brightNamesStars = bstars.bstars
        kdtree = spatial.KDTree([[r[2],r[3]] for r in brightNamesStars])

    if not os.path.exists(fn) or not os.path.exists('./static/cap.corr'):
        return None,"no solved file found"
    with Image.open(fn) as img:
        # image brightness enhancer
        enhancer = ImageEnhance.Brightness(img)
        factor = 3  # gives original image
        im_output = enhancer.enhance(factor)
        
        fitsTable,header= fitsio.read('./static/cap.corr',header=True)
        magstring = 'MAG_VT'
        for hh in header:
            if 'TTYPE' in hh:

                if magstring == header[hh]:
                    break
                elif 'j_mag' == header[hh]:
                    magstring = 'j_mag'
                    break

        astars = []
        for s in fitsTable:
            astar = {}
            astar['RA'] = s['index_ra']
            astar['DEC'] = s['index_dec']

            if s[magstring]>5.9:
                name = 'M%4.3lf'%(s[magstring])
            else:
                name = getStarName(astar['RA'],astar['DEC'],kdtree)
            #print(name, 'ra:', s['index_ra'],s['index_dec'], magstring, s[magstring])
            if name is None:
                continue

            astar['name'] = name
            astar['x'] = s['index_x']
            astar['y'] = s['index_y']
            astar['MAG'] = s[magstring]
            astar['FLUX'] = s['FLUX']
            astar['Background'] = s['BACKGROUND']

            # print('RA: %6.3lf Dec:%6.3lf x:%6.3lf y:%6.3lf Mag:%4.1lf Flux:%d Background:%d'%\
            #     (s['index_ra'],s['index_dec'],\
            #     s['index_x'], s['index_y'],\
            #     s['MAG_VT'],\
            #     s['FLUX'],s['BACKGROUND']))

            astars.append(astar)
        
        
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        draw = ImageDraw.Draw(im_output)

        stars = []
        for star in astars:
            x = star['x']
            y = star['y']
            magR = 4 * (6 - star['MAG'])
            if star['MAG'] < 5:
                draw.ellipse((star['x']-magR, star['y']-magR, star['x']+magR, star['y']+magR),\
                 outline='blue', width=3)

            #print('corrStar',corrStar)
            name = star['name']
            if name is None:
                name = 'M%5.2lf'%(star['MAG'])
            consts = ''
            # find the constelation from the star name
            for c in constellations.constellationNames:

                if c.casefold() in name.casefold():
                    consts = c
                    break
            draw.text((x+10,y-10),name,font = unicode_font)
            #if star['MAG'] > 7.5:
                #draw.ellipse((star['x']-20, star['y']-20, star['x']+20, star['y']+20),\
                 #outline='yellow')
            mag = star['MAG']
            flux = star['FLUX']
            if star['MAG'] < 5:
                txt = 'Mag:%4.1lf FL:%3.0lf BG:%d'%(mag, flux, star['Background'])
                draw.text((x-40,y+20), txt,font = unicode_font)


            if sourceOfImage is not None:
                fn = sourceOfImage
            stars.append({'name': name, 'mag': float(mag), 'flux':float(flux),
                    'constellation': consts,'fileName':fn,
                    'Background':float(star['Background']),'xy':[star['x'],star['y']] }) 

        return stars, im_output ,img.width, img.height
from scipy.optimize import curve_fit
def selectStarsBetweenFluxValues(stars, minflux,maxflux):
    useThese = []
    # find the brightest mag that has a flux less than maxflux
    for s in sorted(stars, key = lambda st: st['mag'], reverse=True):
        flx = s['flux'] 
        if flx > maxflux or flx < minflux:
            continue
        m = s['mag']
        if m < 8:
            useThese.append([m,flx, s['xy'], s['name']])
    return useThese

#make the actual mag versus flux curves (expected and actual)
# given a list of stars and a matplotlib axes
def plotMagcurve(useThese, ax, leg = True):

    mag = [row[0] for row in useThese]
    flux = [row[1] for row in useThese]
    avgflux = sum(flux)/len(flux)
    fig = plt.gcf()
    minMag = min(mag)
    maxMag = max(mag)
    fig.set_facecolor('#404040')

    for ndx, m in enumerate(mag):
        ax.scatter(m,flux[ndx])
        if flux[ndx] > avgflux:
            ax.text(m-.01*m,flux[ndx],useThese[ndx][3], dict(ha='right', va='center', fontsize=8, color='k'))
        #print(s, '%5.3lf'%(stars[s][0]), stars[s][1:])

    ax.grid(color='darkred', linestyle=':')

    def func(x,a,b,c):
        return a * (x ** 2.512) + b * x + c

    #make best fit curve
    params, _ = curve_fit(func, mag, flux)
    a, b, c = params[0], params[1], params[2]
    xes = np.arange( minMag,maxMag, .2)
    yfit1 = [a*pow(x,2.512)+b*x+c for x in xes]

    #pick a point that the median of the magnitude
    x0 = (maxMag + minMag)/2

    fluxMid = a * pow(x0,2.512)+b * x0 + c
    #find the closest real star to this point
    ndx = findClosestPoint(x0,fluxMid,mag,flux)
    
    # now use one of the measured stars as a reference
    x0 = mag[ndx]
    fluxMid = a * (x0 ** 2.512) + b * x0 + c
    while fluxMid < 0:
        ndx += 1
        x0 = mag[ndx]
        fluxMid = a * (x0 ** 2.512) + b * x0 + c
    #make the pefect curve
    cnt = len(yfit1)

    x = []
    y = []


    ax.scatter(x0,fluxMid, color='white')
    for m in np.arange(minMag,maxMag, .2):
        x.append(m)
        fluxPerfect = fluxMid/ (2.512 ** (m-x0))
        y.append(fluxPerfect)


    ax.xaxis.label.set_color('silver')
    ax.yaxis.label.set_color('silver')
    ax.tick_params(colors='silver', which='both')
    ax.set_facecolor('#a0a0a0')

    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Flux')
    params, _ = curve_fit(func, x, y)
    a1, b1, c1 = params[0], params[1], params[2]
    
    if leg:
        label = 'Per a:%4.1lf b:%4.1lf c:%4.1lf'%(a1,b1,c1)
    else:
        label = ''

    qualMetric = math.sqrt(((a1 -a) ** 2 + (b1 -b) ** 2 )/2.)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    ax.plot(x,y,':',label = label)
    ax.plot(xes,yfit1, label='act a:%4.1lf b:%4.1lf c:%4.1lf'%(a,b,c))
    legend = ax.legend(facecolor='k',   framealpha = 0)
    #ax.setp(legend.get_texts(), color='grey')


    #compute the diff from perfect to actual

    for s in useThese:
        idealFlux = (fluxMid)/pow(2.512, s[0] - x0)
        delta = s[1]/idealFlux
        s[2].append(delta)

    #Show position of reference star
    #return reference star and stars with their intensity delta compared with reference
    return {'ref':useThese[ndx][2],'stars':useThese, 'QualityMetric':qualMetric} 

# select stars to be plotted in the flux/mag curves
def plotStarMags(stars, maxflux = 200, minflux = .01):

    starsAvg = {}
    if len(stars) < 6:
        raise ValueError('not enough stars found')

    rawFlux = []
    degamad = []
    # remove supposed gama correction
    fluxMin = 1000
    for s in stars:
        f = s['flux']/255
        rawFlux.append(f)
        degamad.append (s.copy())
        flx = 255 * (f ** 2.2)
        degamad[-1]['flux'] = flx
        if flx < fluxMin:
            fluxMin = flx


    xys = []
    numberofstars = 0
    while True:
        useThese = selectStarsBetweenFluxValues(degamad, minflux,maxflux)
        numberofstars = len(useThese)
        if numberofstars < 5:
            maxflux = maxflux + 50
            if maxflux > 250:
                raise ValueError('not enough stars found')
        else:
            break
    #show histogram of fluxes
    starFlux = [v[1]for v in useThese]

    ax = plt.gca()
    hist = ax.inset_axes([.65,.6,.3,.2])
    n, bins, patches = hist.hist(starFlux, 50, facecolor='b')
    hist.set_facecolor('#808080')
    hist.set_xlabel('flux')
    hist.set_title('flux histo')
    
    result = plotMagcurve(useThese, ax)
    plt.title("Quality:%6.1lf  File: %s"%(result['QualityMetric'],stars[0]['fileName'].split('/')[0]))
    return result


def makeStarStats(star):

        return {'name':star['name'],'mag':star['mag'], 'brightness': star['flux'],\
             'fileName': star['fileName'],'background':star['Background']}

def linkCss():
    return  '<style>'\
        'a:link {'\
        'color: pink;'\
        'background-color: transparent;'\
        'text-decoration: none;'\
        '}'\
        'a:visited {'\
        'color: grey;'\
        'background-color: transparent;'\
        'text-decoration: none;'\
        '}'\
        'a:hover {'\
        'color: red;'\
        'background-color: transparent;'\
        'text-decoration: underline;'\
        '}'\
        'a:active {'\
        'color: yellow;'\
        'background-color: transparent;'\
        'text-decoration: underline;'\
        '}'\
        '</style>'\

def getAllOccurance(starList ,transDB):

    Q = Query()
    occlist = []
    seen = set({})
    for star in starList:
        name = star['name']
        if name in seen:
            continue
        seen.add(name)

        occurances = transDB.search(Q.name == name)
        starstats = makeStarStats(star)

        files = [{'filename':star['fileName'], 'flux':star['flux'], 'background':star['Background']}]
        for otherStar in occurances:
            if otherStar['fileName'] == star['fileName']:
                continue
            files.append({'filename':otherStar['fileName'],'flux':otherStar['flux'],'background':otherStar['Background']})
        occlist.append([name, star['mag'], files])


    sortedStars = sorted(occlist, key= lambda s: s[1] )
    html = linkCss() + '<table border = "2"><th>Star</th><th>Magnitude</th><th colspan="5">Flux : Backgr</th>'
    for star in sortedStars:
        html += '<tr><td>%s</td><td>%3.1lf</td>'%(star[0], float(star[1]))

        for ndx, r in enumerate(star[2]):
            html += '<td><a href="/showImage?fn=%s.jpeg">%3.1lf : %3.0lf</a></td>'%(r['filename'],r['flux'], r['background'])
        html += '<tr>'
    html += '</table>'
        
    return html



    
if __name__ == '__main__':

    path = '/home/pi/work/skysolve/static/'  #only used for debugging.
    results, im, width, height = findStarMags(path + 'cap.jpeg')
    transDB = TinyDB(os.path.join(path, "transparent.json"),indent=4)
    #getAllOccurance(results,transDB)
    try:
        plt.clf()
        results = plotStarMags(results,maxflux = 150)
        draw = ImageDraw.Draw(im, 'RGBA')
    except TypeError as e:
        print(e)
        quit()


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

    im.show("stars")
    #dplotRatios(results)
    plt.show()
    

"""

magnitude m   |  0   1   2   3   4   5   6   7   8    9    10
----------------------------------------------------------------
relative      |  1  2.5 6.3 16  40  100 250 630 1600 4000 10,000
 brightness   |
  ratios      |
(Note that the lower row of numbers is just (2.512)^m.)



(m1 -m2) = 2.5 * LOG(I2/I1)
2.5log(k) = (m1-m2)
log(k) = (m1-m2)/2.5
k = 10 ^ (m1-m2)/2.5
I2/I1 = 10 ^ (m1-m2)/2.5
"""

