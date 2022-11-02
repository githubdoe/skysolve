from re import X
from tinydb import TinyDB, Query
import os
import pprint
import Quality 
from scipy import stats
dbName = 'transparent.json'
def getTransperancyData(path):
    global dbName
    return TinyDB(os.path.join(path,dbName))


#solving /home/pi/work/skysolve/static/history/08_14_22_22_24_16.jpeg
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import numpy as np
import math
if __name__ == '__main__':
    mag = np.arange(1,7, .1)
    flux = list(map(lambda mag: 40 +255/(mag ** 2.512),mag))
    gamaflux = list(map(lambda f: 255*((f/255) ** .4545)-67, flux))
    plt.plot(mag,flux, label = "linear sensor")
    plt.plot(mag,gamaflux, label = 'gamma 2.2')
    plt.legend()
    plt.xlabel('Magnitude')
    plt.ylabel('flux')
    plt.show()
    quit()

    # def makecurve( a,b,c, mag):
    #     flux = []
    #     for m in  mag:
    #         flux.append( a * m ** 2.512 + b * m + c)
    #     return flux

    # def gam(x):
    #     return 255 * ((x/255) ** (1./2.2))

    # def dgeam(x):
    #     return 255 * ((x/255.) ** (2.2))

    # mag = np.arange(0, 7, .2)
    # yg =  makecurve(2.5,-50,350, mag)



    # #plt.plot(x,yg)
    # plt.plot(mag,yg)

    # plt.grid()
    # plt.show()



    path = '/home/pi/work/skysolve/static/'  #only used for debugging.
    db = getTransperancyData(path)
    print("db",db) 
    q = Query()

    samples = db.search(q.sample.matches('.*'))
    rows =2
    cols = 4
    last = len(samples)
    plt.clf()
    group =0
    unicode_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    while group < len(samples):
        plt.clf()
        plt.gcf().set_size_inches(20, 10, forward=True)
        plt.gcf().set_dpi(200)
        plt.rcParams['figure.constrained_layout.use'] = True
        for ndx, s in enumerate(samples[group: group +  2 * rows]):
            col = ndx % 2
            row = int(ndx / 2)
            fn = os.path.join('static','history', samples[ndx + group]['sample'])+'.jpeg'
            with Image.open(fn) as img:
                # image brightness enhancer
                enhancer = ImageEnhance.Brightness(img)
                factor = 8  # gives original image
                im_output = enhancer.enhance(factor) 


            print(group + ndx,samples[group + ndx]['sample'])
            stars = db.search(q.fileName == samples[group + ndx]['sample'])

            if len(stars) > 3:
                try: 
                    plt.subplot(rows,cols, 2 * ndx+ 1 )
                    results = Quality.plotStarMags(stars, maxflux =200, minflux = .01)

                except ValueError as e:
                        continue

                draw = ImageDraw.Draw(im_output, 'RGBA')
                x = results['ref'][0]
                y = results['ref'][1]
                draw.ellipse((x - 20, y-20,x+20,y+20), outline="yellow")
                for s in results['stars']:
                    x = s[2][0]
                    y = s[2][1]
                    if s[2][2] < 5 :
                        draw.text([x,y], s[3], font = unicode_font)

                    if (s[2][2]< 1.25 and s[2][2] >= 1)  or (s[2][2] < 1 and s[2][2] > .75):
                        continue
                    if s[2][2] > 1:
                        delet = s[2][2]
                        color = (10,255,0,80)

                    else:
                        color = (255,0,0,80)
                        delet = 1/s[2][2]


                    rad = 6 +  2 * math.log2(delet)
                    draw.ellipse((x-rad,y-rad,x+rad,y+rad), fill = color)

                plt.subplot(rows, cols, 2 * ndx + 2)
                plt.imshow(im_output)
                plt.title(stars[0]['fileName'])

        filename = 'qualityPlot'+datetime.now().strftime("%m_%d_%y_%H_%M_%S.png") 
        plt.savefig(os.path.join('static','plots', filename ),facecolor='#606060', dpi=200)
        #plt.show()
        group += 4




    # a = list(db.search(q.name.matches('.*')))
    # st = sorted(a, key = lambda s: s['mag'])
    # starsAvg = {}
    # for s in st:
    #     m = s['mag']
    #     if m > 7.5:
    #         continue
    #     flux = s['flux']
    #     name = s['name']
    #     if name  not in starsAvg:
    #         starsAvg[name] = {'mag':m,'flux':[flux]}
    #     else:
    #         starsAvg[name]['flux'].append(flux)
    # mag = []
    # flux = []
    # for s in starsAvg:
    #     starsAvg[s]['avg'] = sum(starsAvg[s]['flux'])/len(starsAvg[s]['flux'])
    #     mag.append(starsAvg[s]['mag'])
    #     flux.append(starsAvg[s]['avg'])


    # plt.scatter(mag,flux)
    # def func(x,a,b,c):
    #     return a * x ** 2.512 + b * x + c
    # params, _ = curve_fit(func, mag, flux)
    # a, b, c = params[0], params[1], params[2]
    # xes = np.arange( min(mag),max(mag), .2)
    # yfit1 = [a*pow(float(x),2.512)+b*float(x)+c for x in xes]

    # plt.plot(xes,yfit1, label='a:%4.2lf b:%4.2lf c:%4.2lf'%(a,b,c))

    # x = []
    # y = []

    # x0 = mag[50]
    # flux0 = flux[50]

    # for m in np.arange(1,7, .2):
    #     x.append(m)
    #     y.append(flux0/ pow(2.512, m-x0))
    # ax = plt.gca()
    # ax.set_facecolor('#202020')
    # plt.plot(x,y,':',label='Perfect')
    # plt.legend()
    # plt.show()


    quit()

    starnames = set({})
    maxavg = 0
    filename = ''
    star = None
    for s in a:
        if s['avg'] > maxavg:
            maxavg = s['avg']
            filename = s['fileName']
            star = s


    for star in a:
        starName = star['name']
        starnames.add(starName)
    print(len(a), len(starnames))
    stats = []
    for star in starnames:
        starSamples = db.search(images.name == star)
        stat = {'name': star, 'mag': starSamples[0]['mag'],'samples':[]}

        for sample in starSamples:
            stat['samples'].append(round(sample['flux']))
        stats.append(stat)

    so = sorted(stats, key = lambda s: s['mag'])
    for star in so:
        print('\n',star['name'], star['mag'], end=' ')
        for s in sorted(star['samples']):
            print(s,end=' ')

        
        

