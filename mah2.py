#!/usr/bin/python
import logging
from hbtep.plot import (red_green_colormap, get_color, sxr_rainbow_colormap, 
    sxr_rainbow_colormap_w)
from matplotlib.colors import Normalize
from hbtep.misc import init_logging
from argparse import ArgumentParser
import subprocess
import numpy as np #numerical package
import pylab #a matlab-similar code, includes plotting support
#standard MDSplus package
from collections import OrderedDict
import sys
sys.path.append('/opt/hbt/python/hbtep')
import tokamac
import pylab as plt
import MDSplus #standard MDSplus package
sys.path.append('/home/byrne/Thesis_Work/mode_analysis')
from subtraction_mode_analysis import crash_finder
sys.path.append('/home/byrne/repos/hbt/python/hbtep')
from mdsplus import *
import miscmod
sys.path.append('/home/byrne/Thesis_Work/tree_reading_scripts')
import Sensors
import Pull_data
import scipy.linalg
from scipy.interpolate import UnivariateSpline as unvspl


#How Mah2 works

#timebase is the timebase of a random feedback sensor.

#OH Bias Start is treated as start time.

#Then the FB,PA,and TA data, R&P is taken from the tree, ignoring any dead 
#sensor or selected exclusions.

#All this data is then smoothed.  Either polynomially, or boxcar.

#The smoothed signal is subtracted.  You're left with the fluctuations.

#Creates a matrix that has each sensor times the number of samples.

#Then every array is stacked together.

#The BD is then performed and then the output is plotted however you choose

log = logging.getLogger('')


def parse_args(args=None):
    '''Parse command line'''

    parser = ArgumentParser()

    parser.add_argument("shotnum", metavar='<shot number>', type=int,
                      help="Shots to plot")

    parser.add_argument("--start_time", metavar='<BD start time, in ms>', 
                        type=float, default=2,
                        help="Start the BD at this time. Default %(default)ms")

    parser.add_argument("--end_time", metavar='<BD end time, in ms>',
                        type=float, default=3,
                        help="End the BD at this time. Default %(default)ms")

    parser.add_argument("--modes", nargs='*',metavar='<mode numbers, starting with zero>',
                        type=int, default=None,
                        help="Study only these modes Default %(default)")

    parser.add_argument("--modenum", metavar='<integer number>', type=int, 
                        default=5, help="The the number of modes to look for. Default (default)")

    parser.add_argument("--nocorrelate", action='store_true', default=False,
                      help="Code scans through the bd and locates coupled modes, select this to *not* do that, to save computational time")

    parser.add_argument("--corplot", action='store_true', default=False,
                      help="Display the Time Domain correlation of each mode to each")
    
    parser.add_argument("--method", metavar='<smoothing method>', type=str,
                        default='boxcar', help="either boxcar or poly")
            
    parser.add_argument('--ignore',metavar='<time window to NOT smooth over>',
                        type = list, default=None,
                        help="choose a time range to be ignored in polynomial smoothing - most useful for RMPs or other slow changing, but non-equilibrium effects")

    parser.add_argument('--exclude',metavar='<sensors to ignore>',type=str,
                        default=None,help="either odd or even, currently")

    parser.add_argument("--debug", action="store_true", default=False,
                      help="Activate debugging output")
    
    parser.add_argument("--quiet", action="store_true", default=False,
                      help="Be really quiet")

    options = parser.parse_args(args)

    return options

def blacklist(shotnum,bad_eq_good_fluct = False):
    to_be_ignored = np.array([])

    #####BAD#####
    #all sensors checked in shots 70000 and 80000.  bad in both
    to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB06_S2P')
    to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB03_S4R')

    #low signal seen in shot 80000 (PA2_S14P) high signal after 85132 (PA2_S27P)
    #strange 'saddleback' (PA1_S16P)
    #all 3 return reasonable looking fluctuations post subtraction
    if not bad_eq_good_fluct:
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA2_S14P')
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA2_S27P')
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA1_S16P')
    else:
        print('\n including sensors PA1_S16P, PA2_S14P, and PA2_S27P.  Equilibrium is bad, but fluctuations seem ok \n')
    #####Were Good, now Bad?#####
    #? zero signal in shot 80000
    to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA1_S29R')

    #bad integration?  signal seen at startup is too low. shot 80000
    #to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA1_S16P')
    #####Were Bad, now Good?#####
    if shotnum < 76732:
        #bad on shot 70000 but looks good in shot 80000
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:PA1_S01P')

        #bad integrator? low signal in 70000, fine by 80000
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB07_S4P')

        #zero signal in 70000, fine by 80000
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S2P')
        
        # signal seems to saturate at 200 Gauss in shot 70245 & 46, with a
        #jump discontinuity in signal at 6ms
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S4P')

        #lots of noise in 70000, seem fine by 80000
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S3P')
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S4P')

        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S1R')
        to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB10_S2R')


    #####Were They Ever Bad?#####
    #? This signal does not look bad in shot 80000, looks fine in 70000
    #to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:TA03_S3P')

    #? Not sure, nothing seems wrong, within the variability of radial sensors
    #to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB08_S3R')
    #to_be_ignored = np.append(to_be_ignored,'sensors.magnetic:FB04_S3R')

    return(to_be_ignored)

def pickup_sensors(shotnum,exclude = None,bad_eq_good_fluct=False):
    to_be_processed = []
    group = OrderedDict()
    group_index = OrderedDict()
    for i in range(1,5):
        group_index['FBP'+str(i)] = []
        group_index['FBR'+str(i)] = []
        group['FBP'+str(i)] = []
        group['FBR'+str(i)] = []

    group['TAP'] = []
    group['TAR'] = []
    group['PA1P'] = []
    group['PA2P'] = []
    group['PA1R'] = []
    group['PA2R'] = []

    group_index['TAP'] = []
    group_index['TAR'] = []
    group_index['PA1P'] = []
    group_index['PA2P'] = []
    group_index['PA1R'] = []
    group_index['PA2R'] = []
    blist = blacklist(shotnum,bad_eq_good_fluct)

    print('bad sensors:')
    print(blist)
    j = 0
    if exclude == None:
        FB_skip = np.array([])
    elif exclude == 'odd':
        FB_skip = np.arange(1,42,2)
    elif exclude == 'even':
        FB_skip = np.arange(1,42,2)
    elif exclude == 'outboard':
        FB_skip = np.arange(0,42)
    else:
        print 'problem with exclude!!!'
######### first FBP #############################
    for i in range(40):
        name = 'sensors.magnetic:FB{0:02}_S{1:1}P'.format(1+i%10,1+i//10)
        if ( (name == blist).any() or ((i+1) == FB_skip).any() ):
            print('Skipping '+name)
        else:
            to_be_processed = np.append(to_be_processed,name)
            group['FBP'+str(1+i//10%4)].append(i%10)
            group_index['FBP'+str(1+i//10%4)].append(j)
            j+=1
        
    name = 'sensors.magnetic:PA'
    PA_rigid = np.append(np.arange(7),np.arange(25,32))
######### First PAP #############################
    if ((exclude == 'odd') or (exclude == 'outboard')):
        include_sensors = PA_rigid
        print "Skipping PA1 shell mounted only!"
    else:
        include_sensors = range(32)
        
    for i in include_sensors:
        if (name+'1_S{0:02}P'.format(i+1) != blist).all():
            to_be_processed = np.append(to_be_processed,name+'1_S{0:02}P'.format(i+1))
            group['PA1P'].append(i)
            group_index['PA1P'].append(j)        
            j+= 1
        else: print('Skipping '+name+'1_S{0:02}P'.format(i+1))

    if ((exclude == 'even') or (exclude == 'outboard')):
        include_sensors = PA_rigid
        print "Skipping PA2 shell mounted"
    else:
        include_sensors = range(32)
       
    for i in include_sensors:
        if (name+'2_S{0:02}P'.format(i+1) != blist).all():
            to_be_processed = np.append(to_be_processed,
                                        name+'2_S{0:02}P'.format(i+1))
            group['PA2P'].append(i)
            group_index['PA2P'].append(j)       
            j+= 1
        else: print('Skipping '+name+'2_S{0:02}P'.format(i+1))

######### Third TAP #############################
    for i in range(30):
        name = 'sensors.magnetic:TA{0:02}_S{1:1}P'.format(1+i//3,1+i%3)
        if ( name != blist).all():
            to_be_processed = np.append(to_be_processed, name)
            group['TAP'].append(i+i//3)
            group_index['TAP'].append(j)
            j+= 1
        else: print('Skipping '+name)

######### Fourth FBR #############################
    if exclude != 'radial':
        for i in range(40):
            name = 'sensors.magnetic:FB{0:02}_S{1:1}R'.format(1+i//4,1+i%4)
            if ( (name == blist).any() or ((i+1) == FB_skip).any() ):
                print('Skipping '+name)
            else:
                to_be_processed = np.append(to_be_processed,name)
                group['FBR'+str(1+i//10%4)].append(i%10)
                group_index['FBR'+str(1+i//10%4)].append(j)
                j+=1

######### Fifth PAR #############################
        name = 'sensors.magnetic:PA'
        if ((exclude == 'odd') or (exclude == 'outboard')):
            include_sensors = PA_rigid
        else:
            include_sensors = range(32)

        for i in include_sensors:
            if (i+1)%2 == 1:
                if (name+'1_S{0:02}R'.format(i+1) != blist).all():
                    to_be_processed = np.append(to_be_processed,
                                                name+'1_S{0:02}R'.format(i+1))
                    group['PA1R'].append(i)
                    group_index['PA1R'].append(j)       
                    j+= 1
                else: print('Skipping '+name+'1_S{0:02}R'.format(i+1))

        if ((exclude == 'even') or (exclude == 'outboard')):
            include_sensors = PA_rigid
        else:
            include_sensors = range(32)

        for i in include_sensors:
            if (i+1)%2 == 1:
                if (name+'2_S{0:02}R'.format(i+1) != blist).all():
                    to_be_processed = np.append(to_be_processed,
                                                name+'2_S{0:02}R'.format(i+1))
                    group['PA2R'].append(i)
                    group_index['PA2R'].append(j)       
                    j+= 1
                else: print('Skipping '+name+'2_S{0:02}R'.format(i+1))
######### Sixth TAR #############################
        for i in range(30):
            if (i)%3 == 1:
                name = 'sensors.magnetic:TA{0:02}_S{1:1}R'.format(1+i//3,1+i%3)
                if ( name != blist).all():
                    to_be_processed = np.append(to_be_processed,name)
                    group['TAR'].append(i+i//3)
                    group_index['TAR'].append(j)
                    j+= 1
                               
    return(to_be_processed,group,group_index)

def eq_subtract(window, time, signal, method = None,ignore = None,order = 4):
    smthsig = np.zeros(np.shape(time))
    index = np.arange(len(time))

    if ignore != None:
        method == 'poly'

    if method == 'poly':
        if ignore != None:
            index = np.where( ((time<=ignore[0])+(time>=ignore[1]))*
                              ((time>=window[0])*(time<=window[1])))[0]

        else:
            index = np.where((time>=window[0]-.0001)*(time<=window[1]+.0001))[0]

    if method == 'spline':
        print('spline is probably broken by now!!!')
        spl = unvspl(time[index],signal[index])
        smthsig = spl(time)

    elif method == 'poly':
        plt.figure(15)
        plt.plot(time[index],signal[index])
    #went with a 4th order polynomial.  cubic spline is possible.
        z = np.polyfit(time[index],signal[index], order)
        for i,coeff in enumerate(z):
            smthsig += coeff*time**(order-i)
        plt.plot(time,signal-smthsig)
    else:
        #doing a double boxcar smooth of one period of an 8kHz mode
        #anything equal or faster should remain, anything slower should
        #be removed - this includes RMP's!!! 
        #Anyway, 62.5 samples is one period. Testing 3 passes with a pure 8kHz 
        #tone gives an attenuation of 5 orders of magnitude, 
        #4kHz, slightly more than 75%
        #1kHz (RMP frequency) less than 10%
        N = 65.
        pass1 = np.convolve(signal,np.ones(N)/N,mode = 'same')
        pass2 = np.convolve(pass1,np.ones(N)/N,mode = 'same')
        smthsig = np.convolve(pass2,np.ones(N)/N,mode = 'same')

    return(smthsig)
    #return(spl)

def get_flucts(shotnum,t0,t1,sensors,method = None,ignore = None,order =4,bdplot = True):
    '''This creates an array of the sensor signals, in groups,
    in order, FB,PA1,PA2,TA, Poloidal first, then Radial
    '''
    print('Pulling the data from the tree & extracting fluctuations...\n')
    tree = MDSplus.Tree('hbtep2',shotnum)

    tbrkdwn = max(
        1e-6*(tree.getNode('timing.banks:oh_el').data()+100),
        1e-6*(tree.getNode('timing.banks:vf_el').data()+100))

    drtn = [-.001,.013]
    (t,ip) = Pull_data.pull_data(tree, '.sensors.rogowskis.ip', 
                                     zero = True, duration = drtn)

    try:
        (t,sh) = Pull_data.pull_data(tree, '.sensors.sh_current', 
                                     zero = True, duration = drtn)
    except:
        print('SH_CURRENT node was not in tree for this shot')
        sh = ip *0.0

    #MR = miscmod.calc_r_major(tree,times=t, byrne = True)
    (q,MR) = miscmod.calc_q(tree, t, byrne = True)

    tdsrpt = crash_finder(shotnum,t)[1]

    print('loading data from tree...')
    Vresp = []
    for i,sens in enumerate(sensors):

        (time,signal) = Pull_data.pull_data(tree, sens, zero = True,
                                            duration = drtn)
           
        Vresp.append(signal)
    print('data loaded from tree...')


    if (tdsrpt-.0001)<t1:
        print('Plasma has disruption before t = '+str(t1*1000)+' ms\n')
        print('Truncating BD window to {0:01.2f}ms / {1:01.2f}ms'.format(
                t0*1e3,(tdsrpt-.0001)*1e3))
        newEndIndex = np.argmin(abs(t-(tdsrpt-.0001)))
        t1 = t[newEndIndex]

    if tbrkdwn>t0:
        print('Plasma has breakdown after t = '+str(t0*1000)+' ms\n')
        print('Truncating BD window to {0:01.2f}ms / {1:01.2f}ms'.format(
                tbrkdwn*1e3,t1*1e3))
        newStartIndex = np.argmin(t-(tbrkdwn))
        t0 = t[newStartIndex]

    if method == 'poly':
        #poly fitting wants slow change as possible, crop breakdown and disruption
        drtn = [tbrkdwn+.0001,tdsrpt-.0001]

    read_index = np.where((t>=drtn[0])*(t<=drtn[1]))[0]
    print('Smoothing signal from {0:01.2f}ms to {1:01.2f}ms...'.format(
            t[read_index][0]*1e3,t[read_index][-1]*1e3))

    flucts = []
    index = np.where( (t>=t0)*(t<=t1) )[0]

    for i,sens in enumerate(sensors):      
        smth = eq_subtract(drtn,time,Vresp[i],method,ignore,order)

        flucts.append(Vresp[i] - smth)
 
    
    #plt.figure(1)
    #plt.title("sample sensor before and after eq subtraction:\n" + sensors[1])
    #plt.plot(time,Vresp[1])
    #plt.plot(time,flucts[1])
    #plt.fill_between([t0,t1],-0.005,.03,facecolor = 'red',alpha = .5)
 
    #plt.figure(2)
    #plt.title("sample sensor before and after eq subtraction:\n" + sensors[98])
    #plt.plot(time,Vresp[98])
    #plt.plot(time,flucts[98])
    #plt.fill_between([t0,t1],-0.005,.03,facecolor = 'red',alpha = .5)

    index = np.where( (time>=t0)*(time<=t1) )[0]
    if bdplot:
        fig = plt.figure(12,figsize = (30,11))
        ax = fig.add_subplot(3,3,1)
    #plt.contourf(time,np.arange(1,33),Vresp[:32],100)
        plt.plot(t,ip/1e3,'--',label = 'ip')
        plt.plot(t,sh/1e3,'--',label = 'shaping')
        plt.plot(t[read_index],ip[read_index]/1e3,'b')
        plt.plot(t[read_index],sh[read_index]/1e3,'g')
        plt.legend(loc = 2)
        plt.ylabel('Current (kA)')
        plt.fill_between(time[index],0,2e4,facecolor = 'red',alpha = .5)
        plt.xlim(tbrkdwn-.001,tdsrpt+.001)
        plt.ylim(0,20)

        ax = fig.add_subplot(3,3,4)
        plt.ylabel('MR (cm)')
        plt.plot(t[read_index],MR[read_index]*100)
        for i in range(16):
            plt.plot(t,np.zeros(len(t))+88.5+i*.5,':k')
        plt.fill_between(time[index],87,97,facecolor = 'red',alpha = .5)
        plt.ylim(88,96)
        plt.xlim(tbrkdwn-.001,tdsrpt+.001)

        ax = fig.add_subplot(3,3,7)
        plt.ylabel('q*')
        plt.plot(t[read_index],q[read_index])
        plt.fill_between(time[index],0,5,facecolor = 'red',alpha = .5)
        plt.plot(t,np.zeros(len(t))+3,'--k')
        plt.ylim(2,4)
        plt.xlim(tbrkdwn-.001,tdsrpt+.001)
    #plt.contourf(time,np.append(np.arange(17,33),np.arange(1,17)),
    #             np.append(flucts[16:32],flucts[:16],axis=0),100)
 
    return(time,index,MR,np.asarray(Vresp),np.asarray(flucts))

def BD(flucts,index):
    print "BD!"
    svd = scipy.linalg.svd(flucts[:,index],full_matrices =0)
    
    return(svd)

def correlation(dt,group_index,bd,nummodes=6,modes = None,
                plot = False,exclude = None):
    if modes:
        nummodes = 2*len(modes)//2+2*(len(modes)%2)
    else:
        modes = np.arange(nummodes)
    pairs = []
    correlation = []
    c = []
    cfft = []
    skip = np.array([])
    bestmatch = np.zeros((nummodes),dtype = int)
    pairmatch = np.zeros((nummodes))
    pairfreq = np.zeros((nummodes))

    if exclude != 'even':
        m_array = 'PA2P'
    else:
        m_array = 'PA1P'

    mfreq = np.fft.fftfreq(len(group_index[m_array]),
                           1./len(group_index[m_array]))

    if exclude != 'outboard':
        n_array = 'FBP3'
    else:
        n_array = 'TAP'
    print n_array
    nfreq = np.fft.fftfreq(len(group_index[n_array]),
                           1./len(group_index[n_array]))
    timefreq = np.fft.fftfreq(len(bd[2][0]),dt)
    if plot:    
        fig = plt.figure(14,figsize = (20,12))
        fig.subplots_adjust(left = .040, right = 1, top = .90, 
                            bottom = .075,hspace = .15,wspace = .15)
        height = np.ceil(np.sqrt(nummodes))
        width = np.ceil(nummodes/height)
        colors = ['b','g','r','m','y','c']
        symbols = ['o','v','s','*','+','D']
        colorsym =[]
        for i in range(nummodes**2):
            if i%nummodes != i//nummodes:#self correlation won't be used
                if i//nummodes > i%nummodes:
                    colorsym.append(colorsym[i%nummodes*nummodes+i//nummodes])
                else:
                    colorsym.append(colors[min(i//nummodes%6,i%nummodes%6)]+
                                    symbols[max(i//nummodes%6,i%nummodes%6)])
            else: colorsym.append('bo')

    length = len(bd[2][0])
    freq = 1/(4*np.arange(1,min(length,50))*dt)
    

    for k in range(nummodes**2):
        last_mode = k//nummodes - 1
        this_mode = k//nummodes
        match_mode = k%nummodes

        if match_mode == 0:
            correlation.append([])
            c.append([])
            cfft.append([])
            if plot:
                ax = fig.add_subplot(height,width,this_mode+1)
                plt.title('Mode correlations to mode # {0:1d}'.format(modes[this_mode]+1))

        correlation[this_mode].append([])
        c[this_mode].append([])
        cfft[this_mode].append([])

        if this_mode != match_mode:
            x = np.fft.fft(bd[2][modes[this_mode]])
            x = np.sqrt(x.real**2+x.imag**2)
            y = np.fft.fft(bd[2][modes[match_mode]])
            y = np.sqrt(y.real**2+y.imag**2)
            exp_x = x-np.mean(x)
            exp_y = y-np.mean(y)
            cfft_pearson_num = np.sum(exp_x*exp_y)
            cfft_pearson_denom = np.sqrt(np.sum(exp_x**2)*np.sum(exp_y**2))
            cfft_pearson = cfft_pearson_num/cfft_pearson_denom
            cfft[this_mode][match_mode].append(abs(cfft_pearson))
            #correlate the two and plot the values
            
            for i in range(1,length):
                print i,length
                ln = length-i
                a = bd[2][modes[this_mode]][0:ln]
                b = bd[2][modes[match_mode]][i:]
                exp_a = a-np.mean(a)
                exp_b = b-np.mean(b)
                c_pearson_num = np.sum(exp_a*exp_b)
                #print i,k,ln*np.sum(x)**2 - np.sum(x**2)
                #print ln*np.sum(y)**2 - np.sum(y**2)
                #c_pearson_denom = ((np.sqrt(ln*np.sum(x)**2 - np.sum(x**2)) *
                #                   np.sqrt(ln*np.sum(y)**2 - np.sum(y**2)) ))
                c_pearson_denom = np.sqrt(np.sum(exp_a**2)+np.sum(exp_b**2))
                c_pearson = c_pearson_num/c_pearson_denom
                c[this_mode][match_mode].append(np.abs(c_pearson))
                #correlation[this_mode][match_mode].append(
                #    np.abs(np.sum(bd[2][modes[this_mode]][0:length-i]*
                #                  bd[2][modes[match_mode]][i:])) /
                #    np.sum(bd[2][modes[this_mode]][0:length-i]**2))
                
            if plot:
                #plt.plot(freq[::-1],correlation[this_mode][match_mode][::-1],
                #         colorsym[k],label = (str(this_mode+1)+'-'
                #                              +str(match_mode+1)))
                plt.plot(freq[::-1],
                         abs(np.asarray(c[this_mode][match_mode][::-1])),
                         colorsym[k],label = (str(this_mode+1)+'-'
                                              +str(match_mode+1)))
                ax.set_xscale('log')
                plt.legend()
                plt.ylim(0,1)

        #at the end of every other round
        if (match_mode == nummodes-1):
            pairmatch[this_mode] = max([item for sublist in 
                                        c[this_mode] for 
                                        item in sublist])
            bestmatch[this_mode] = np.argmax([item for sublist in 
                                              c[this_mode] for 
                                              item in sublist])
            #high = max([item for sublist in c[this_mode] for item in sublist])
            #low = min([item for sublist in c[this_mode] for item in sublist])

            #highfft = max([item for sublist in cfft[this_mode] 
            #               for item in sublist])
            #lowfft = min([item for sublist in cfft[this_mode] 
            #              for item in sublist])

            #if abs(highfft)>abs(lowfft):
                #pairmatch[this_mode] = high
            #pairmatch[this_mode] = max([item for sublist in cfft[this_mode] for 
            #                            item in sublist])
            #bestmatch[this_mode] = np.argmax(cfft[this_mode])
            #else:
                #pairmatch[this_mode] = low
            #    pairmatch[this_mode] = abs(min([item for sublist in 
            #                                    cfft[this_mode] for 
            #                                    item in sublist]))
            #    bestmatch[this_mode] = np.argmin([item for sublist in 
            #                                      cfft[this_mode] for item 
            #                                      in sublist])
            #pairmatch[this_mode] = max([item for sublist in 
            #                            c[this_mode] for item in sublist])
            #bestmatch[this_mode] = np.argmax([item for sublist in 
            #                                  c[this_mode] for item in sublist])
            if bestmatch[this_mode] > 49*(this_mode):
                bestmatch[this_mode] += 49
            pairfreq[this_mode] = freq[bestmatch[this_mode]%49]
            bestmatch[this_mode] = bestmatch[this_mode]//49
            #pairfreq[this_mode] = abs(timefreq[np.argmax(np.sqrt(np.fft.fft(bd[2][modes[bestmatch[this_mode]]]).real**2+np.fft.fft(bd[2][modes[bestmatch[this_mode]]]).imag**2))]*1e-3)
            if bestmatch[this_mode] < this_mode:
                #print 'mode : ', this_mode
                #print 'correlations 1 : ', cfft[this_mode]
                #print 'correlations 2 : ', cfft[bestmatch[this_mode]]
                #print 'modepair : ',bestmatch[this_mode], this_mode
                #print 'match 1 : ', bestmatch[this_mode]
                #print 'match 2 : ', bestmatch[bestmatch[this_mode]]
                if this_mode == bestmatch[bestmatch[this_mode]]:
                    if ((abs(pairfreq[this_mode]/
                             pairfreq[bestmatch[this_mode]] -1) > 0.25) or
                        (abs(abs(pairmatch[this_mode]/
                                pairmatch[bestmatch[this_mode]])-1) > .25)):
                        mode_power = 'Very Weakly Matched'
                    elif (abs(pairmatch[this_mode])+
                          abs(pairmatch[bestmatch[this_mode]]) < 1.25):
                        mode_power = 'Weakly Matched'
                    else:
                        mode_power = 'Strongly Matched'
                    
                    print('{0:s} Pair Found! Modes {1:1.0f} & {2:1.0f}'.format(
                            mode_power, modes[bestmatch[this_mode]]+1,
                            modes[this_mode]+1))
                    nfft1 = np.fft.fft(bd[0][group_index[n_array],
                                             modes[this_mode]])
                    nfft2 = np.fft.fft(bd[0][group_index[n_array],
                                             modes[bestmatch[this_mode]]])
                    mfft1 = np.fft.fft(bd[0][group_index[m_array],
                                             modes[this_mode]])
                    mfft2 = np.fft.fft(bd[0][group_index[m_array],
                                             modes[bestmatch[this_mode]]])
                    navgfft = .5*(np.abs(nfft1)+np.abs(nfft2))
                    mavgfft = .5*(np.abs(mfft1)+np.abs(mfft2))

                    print('Best Guess as to m/n shape: {0:01f}/{1:01f}'.format(
                            mfreq[1:][np.argmax(mavgfft[1:])],
                            nfreq[1:][np.argmax(navgfft[1:])]))
                    print('freqencies {0:2.01f} & {1:3.01f}kHz'.format(
                            pairfreq[this_mode]*1e-3,
                            pairfreq[bestmatch[this_mode]]*1e-3))
                    print('correlation {0:1.02f} & {1:1.02f}'.format(
                                pairmatch[this_mode],
                                pairmatch[bestmatch[this_mode]]))
                    print('% Total Power {0:3.02f}\n'.format(
                            (bd[1][modes[this_mode]]**2 +
                             bd[1][modes[bestmatch[this_mode]]]**2)/
                            (1e-2*np.sum(bd[1]**2) ) 
                            ))

                    pairs.append([bestmatch[this_mode],this_mode])
                    skip = np.append(skip,[this_mode,bestmatch[this_mode]])

    #return(correlation,pairmatch,bestmatch,pairfreq)
    return(pairs)

def phase_shift(phs,sine, cosine):
    '''takes two modes and sums them s.t. they are "shifted" by that many 
    degrees. Useful for matching BD modes to compare how similar they are 
    in shape'''
    phs = phs*(np.pi/180.)
    return(sine*np.cos(phs)+cosine*np.sin(phs) ,
           cosine*np.cos(phs)-sine*np.sin(phs) )

def phase_match(matchMode,compareMode):

    maxCorr = 0
    for i in range(360):
        (newSin,newCos) = phase_shift(i,matchMode[0],matchMode[1])
        corr = ( np.sum(newSin*compareMode)/np.sum(compareMode**2),
                  np.sum(newCos*compareMode)/np.sum(compareMode**2))

    if abs(maxCorr) < np.max([abs(corr[0]),abs(corr[1])]):
        maxCorr = corr[np.argmax([abs(corr[0]),abs(corr[1])])]
        bestShift = i
        if maxCorr <0:
            bestShift += 180

    (bestSin,bestCos) = phase_shift(bestShift,matchMode[0],matchMode[1])
    return(bestSin,bestCos)

def fileScan(fileName = None,lines = None,searchTerm = None):
    pos = []
    if (fileName == None) and (lines == None):
        print('fileScan needs input!')
        return(-1)

    elif (fileName != None) and (lines != None):
        print('Provide fileScan with EITHER a filename or a list of strings!')
        return(-1)

    elif(lines != None) and (searchTerm == None):
        print('Please provide a searchTerm to scan for!')
        return(-1)

    elif fileName !=None:
        f = open(fileName,'r')
        lines = f.readlines()
        f.close()
        
    if searchTerm != None:
        for i,line in enumerate(lines):
            if line.strip() == searchTerm:
                pos.append(i)
        if fileName != None:
            return(pos,lines)
        else:
            return(pos)
    else:
        return(lines)

def fileWrite(fileName,writeInfo,pos= None):
    f = open(fileName,'w')
    for line in writeInfo:
        f.write(line)
    f.close() 

def dcon(n,TokaMac = False,Ip = None,MajR = None, Ish = None,pressure = None,Vf = None):
    ''' Intention with this code is to automate the creation of a plot
    similar to what was used in the HBT-EP narrative for 2011.  I'd like very
    much to make this usable from any directory, with a command to tell the
    code in which directory to run DCON.  Unfortunately my Linux skills aren't
    up to the task.'''
    if TokaMac == True:
        tokamac.clean('.')
        TokInLines = fileScan(fileName = 'TokIn.dat')
        #f = open('TokIn.dat','r')
        #TokInLines = f.readlines()
        #f.close()
        wrt = False
        if ((Vf != None) or (Ip != None) or (Ish != None) or 
            (MajR != None) or (pressure != None)):
            wrt = True
            if Vf != None:
                pos = (1 + fileScan(lines = TokInLines,
                                    searchTerm = 'Name = VF_Coil_Current')[0])
                TokInLines[pos] = '\tValue = -{0:02.3f}e3\r\n'.format(Vf)
            if Ip != None:
                pos = (5+ fileScan(lines = TokInLines,
                                       searchTerm = 'K_Plasma')[0])
                TokInLines[pos] = '\tIp0 = {0:02.2f}e3\r\n'.format(Ip)
                pos = (1+ fileScan(lines = TokInLines,
                                       searchTerm = 'Name = PlasmaCurrent')[0])
                TokInLines[pos] = '\tValue = {0:02.3f}e3\r\n'.format(Ip)
            if Ish != None:
                pos = (2 + fileScan(lines = TokInLines,
                                    searchTerm = 'Name = SH_Coil')[0])
                TokInLines[pos] = '\tInitialCurrent = {0:02.3f}e3\r\n'.format(Ish)
                pos = (1 + fileScan(lines = TokInLines,
                                    searchTerm = 'Name = SH_Coil_Current')[0])
                TokInLines[pos] = '\tValue = {0:02.3f}e3\r\n'.format(Ish)

            if MajR != None:
                pos = (2+ fileScan(lines = TokInLines,
                                   searchTerm = 'K_Plasma')[0])
                TokInLines[pos] = '\tR0 = {0:01.5f}\r\n'.format(MajR)
                if ((MajR >0.92) or (MajR<0.903)):
                    TokInLines[pos+2] = '\ta0 = {0:01.2f}\r\n'.format(min(
                            1.07-MajR,MajR-0.75))
                pos = (1+ fileScan(lines = TokInLines,
                                   searchTerm = 'Name = CorePress')[0])
                TokInLines[pos] = '\tX = {0:01.5f}\r\n'.format(MajR)
                pos = (2+ fileScan(lines = TokInLines,
                                   searchTerm = 'Name = MagAxis')[0])
                TokInLines[pos] = '\tX = {0:01.5f}\r\n'.format(MajR)

            if pressure != None:
                pos = (3+ fileScan(lines = TokInLines,
                                   searchTerm = 'Name = CorePress')[0])
                TokInLines[pos] = '\tValue = {0:03.1f}\r\n'.format(pressure)
        if wrt:
            fileWrite('TokIn.dat',TokInLines,pos= None)
            #f = open('TokIn.dat','w')
            #for line in TokInLines:
            #    f.write(line)
            #f.close()

        #subprocess.call(["TokaMac"])
        tokamac.run_tokamac('/home/byrne/Thesis_Work/mode_analysis/TokaMac/TokaMac_tutorial/')
    shape = 0

    #f = open('TokIn.dat','r')
    #TokInLines = f.readlines()
    #f.close()
    (pos,TokInLines) = fileScan(fileName = 'TokIn.dat', searchTerm = 'K_PsiGrid')
    gridPts = int(TokInLines[pos[0]+1].strip().split()[2])
    TokOutLines = fileScan(fileName = 'TokOut_Plasma.out')
    #filePath = 'TokOut_Plasma.out'#direc+'/dcon.out'
    #f = open(filePath,'r')
    #lines = f.readlines()
    #f.close()
    
    for line in TokOutLines:
        if line.strip():
            line = np.asarray(line.strip().split())
            if (line == 'qStar').any():
                q_star = float(line[np.where(line == 'qStar')[0]+2][0][:-1])
                break

    (X,Z,Psi) = tokamac.get_flux('./')
    (X,Z,Cur) = tokamac.get_current('./')
    (sep,lim)  = tokamac.get_separatrix('./')
    limIndex = np.max(np.where(np.asarray(X)<1.07))
    diverted = False
    if sep != []:
        if TokaMac_diverted(sep[0],(Psi[gridPts/2,limIndex],X[limIndex]),(Psi[gridPts/2,limIndex+1],X[limIndex+1])):
            print '\n We have a DIVERTED plasma! \n'
            diverted = True
    if not diverted:
        print '\n Plasma is LIMITED \n'
        print lim
    lim = lim[0]#[:-1]

    unstableModeEnergy = list()
    MarginalModeEnergy = list()
    unstableModeShape = list()
    MarginalModeShape = list()
    #specify the toroidal mode to look at
    for j,N in enumerate(n):
        #filePathin = 'dcon.in'#direc+'/dcon.in'
        #f = open(filePathin,'r')
        #inputLines = f.readlines()
        #f.close()
        inputLines = fileScan('dcon.in')
        unstableModeEnergy.append(list())
        MarginalModeEnergy.append(list())
        unstableModeShape.append(list())
        MarginalModeShape.append(list())
        if inputLines[6][-2] != str(N):
            inputLines[6] = inputLines[6][:-2]
            inputLines[6] += str(N)+'\n'
            fileWrite('dcon.in',inputLines)
            #f = open(filePathin,'w')
            #for line in inputLines:
            #    f.write(line)
            #f.close()

        subprocess.call(["dcon"])

        #filePathout = 'dcon.out'#direc+'/dcon.out'
        #f = open(filePathout,'r')
        #lines = f.readlines()
        #f.close()
        (pos,outLines) = fileScan(fileName = 'dcon.out',searchTerm = 'q0        qmin        q95       qmax        qa        crnt       I/aB')
        qaDcon = float(outLines[pos[0]+2].strip().split()[2])

        ipDcon = float(outLines[pos[0]+2].strip().split()[5])*1e6

        energyLines = None
        shapeLines = None

        for i,line in enumerate(outLines):
            if line == ' Total Energy Eigenvalues:\n':
                energyLines = i+4
                unstableModeEnergy[j].append(
                    float(outLines[energyLines].split()[3]))
                MarginalModeEnergy[j].append(
                    float(outLines[energyLines+1].split()[3]))
            if line == ' Total Energy Eigenvectors:\n':
                shapeLines = i+8
            if ((energyLines != None) and (shapeLines != None)):
                break

        (mr,MR)  = np.asarray(outLines[12].split()[0:2],dtype = float)


        for i,line in enumerate(outLines[shapeLines:]):
            if len(line.split()) == 6:
                shape = int(line.split()[1])
            if(line.strip() == 'isol  imax   plasma     vacuum     total'):
                shapeLines += i+6
                break

        unstableModeShape[j].append(shape)

        for line in outLines[shapeLines:]:
            if len(line.split()) == 6:
                shape = int(line.split()[1])
                break

        MarginalModeShape[j].append(shape)

    return((X,Z,Psi,sep,lim,diverted,Cur),(ipDcon,MR),(qaDcon,q_star),
           (unstableModeShape,unstableModeEnergy),
           (MarginalModeShape,MarginalModeEnergy))

def dcon_surf(run = False):
    if run:
        subprocess.call(["dcon"])
    #f = open('dcon_surf.out','r')
    #lines = f.readlines()
    lines = fileScan('dcon_surf.out')
    (R_start,Z_start) = (0,0)
    R = np.array([],dtype = float)
    Z = np.array([],dtype = float)
    c = np.array([],dtype = float)
    s = np.array([],dtype = float)

    for i,line in enumerate(lines):
        if line.strip() == 'radial position r:':
            R_start = i+2
            j = R_start
            while lines[j].strip().split():
                R = np.append(R,np.asarray(lines[j].strip().split(),
                                           dtype = float))
                j+=1
            R_done = True
        if line.strip() == 'axial position z:':
            Z_start = i+2
            j = Z_start
            while lines[j].strip().split():
                Z = np.append(Z,np.asarray(lines[j].strip().split(),
                                           dtype = float))
                j +=1
        if (line.strip() == 
            'normal magnetic eigenvector, isol =   1, cos factor:'):
            cos_start = i+2
            j = cos_start
            while lines[j].strip().split():
                c = np.append(c,np.asarray(lines[j].strip().split(),
                                               dtype = float))
                j +=1
        if (line.strip() == 
            'normal magnetic eigenvector, isol =   1, sin factor:'):
            sin_start = i+2
            j = sin_start
            while lines[j].strip().split():
                s = np.append(s,np.asarray(lines[j].strip().split(),
                                               dtype = float))
                j +=1
    return(R,Z,c,s)

def TokaMac_diverted((sepx, sepz, sepflux),(edge_flux1,edge_location1),
                     (edge_flux2, edge_location2)):
    ''' takes the output of TokaMac (sepx,sepz,and sepPsi) as a tuple, and
    the locations of the grid points on either side of the limiter 
    (currently in a 129X129 grid, X[118] and X[119]).  It linearly interpolates
    the flux at the limiter, and ensures that the LCFS (sepPsi) flux is LESS
    than that at the limiter.  Plasma flux is measured as negative, so you drop
    in flux as you go through the center, and rise again as you leave, traveling
    from inboard to outboard.'''

    limiter_x = 1.07227
    grid_space = abs(edge_location2-edge_location1)
    limiter_flux = (edge_flux1*(edge_location2-limiter_x)+
                    edge_flux2*(limiter_x-edge_location1) )/(grid_space)

    if (np.sqrt((sepx-.9032)**2+sepz**2) > 0.15):
        return(False)

    if limiter_flux < sepflux:
        return(False)

    else: return(True)


def stripey(time,group_index,sensor_group,modes=None,flucts = None,bd=None,window = None):
    t_index = np.where((time>=window[0])*(time<=window[-1]))[0]
    t_start = time[t_index][0]
    t_end = time[t_index][-1]
    top_len = len(group_index[sensor_group])
    chron_len = len(time[t_index])

    if (sensor_group == 'PA2P') or (sensor_group == 'PA2P'):
        sensor_start = 180
        #the actual locations of the PA sensors are too complicated to go into
        #sensor_sep = 11.63
        #sensor_offset = 11.63/2.
        sensor_sep = 360/32.
        sensor_offset = 360/64.
        angles = np.arange(sensor_start-360, sensor_start,sensor_sep)+sensor_offset
    elif sensor_group == 'TAP':
        #for the sake of simplicity of plotting, I'm calling TA1_S1P 'zero phi'
        #technically it isn't.  It could be useful to uncomment the lines
        #below to look for error fields.  Not now though.
        #sensor_start = 180+72-9 #TA1P is the trailing sensor in Chamber 1
        sensor_start = 360
        angles = np.zeros(np.shape(group_index[sensor_group]))
        for i in range(10):
            angles[3*i:3*(i+1)] = (i*4+np.arange(3))*9

    if flucts != None:
        stripey = flucts[group_index[sensor_group]][:,t_index]

    elif bd != None:
        topos = bd[0]
        chronos = bd[2]
        mag = [bd[1][modes[0]],bd[1][modes[1]]]
        Top1 = topos[group_index[sensor_group],modes[0]]
        Chr1 = chronos[modes[0]]
        Top2 = topos[group_index[sensor_group],modes[1]]
        Chr2 = chronos[modes[1]]
        stripey = np.outer(mag[0]*Top1,Chr1) + np.outer(mag[1]*Top2,Chr2)

    else:
        print 'include flucts or bd in your call!'
        return
    print np.shape(stripey)
    print top_len
    print chron_len
    print np.shape(stripey[0])
    print np.shape(np.append(stripey.reshape(-1),stripey[0]))
    print (top_len+1)*chron_len
    print np.shape(np.append(stripey.reshape(-1),
                         stripey[0]).reshape(top_len+1,chron_len))
    stripey = (np.append(stripey.reshape(-1),
                         stripey[0].reshape(-1)).reshape(top_len+1,chron_len))
    sig_range = np.abs(stripey.reshape(-1)).max()
    print(len(angles))
    sensorAngles = np.append(angles,angles[0]+360)
    print(len(sensorAngles),np.shape(stripey))
    plt.contourf(time[t_index]*1e3, sensorAngles, 
                 stripey, 500, cmap=red_green_colormap(),
                 norm=Normalize(vmin=-sig_range, vmax=sig_range))
    print(t_start,.125*(t_end-t_start)*1e3)
    plt.plot(np.zeros((len(sensorAngles)))+(t_start+.125*(t_end-t_start))*1e3,
             sensorAngles,'wd')
    #plt.xticks([])


    return(stripey,sensorAngles)

def main(shotnum, t0, t1, modenum = 5, modes = None,bdplot = True,
         corplot = False,exclude = None, ignore = None, correlate = True, 
         method = 'boxcar',order = 4,bad_eq_good_fluct = False,):
    if ((exclude != None) and (exclude != 'even') and (exclude != 'odd') and 
        (exclude != 'outboard')):
        print("'exclude' must be either 'even', 'odd', outboard or omitted!")
        return 0
    #print('inputs:'+str(corplot))
    (sensors, group, group_index) = pickup_sensors(shotnum,exclude,bad_eq_good_fluct)
    
    if ignore != None:
        print('Non-continuous time range!  Polynomial smoothing to order {0:d}'.format(order))

    (time,index,MR,Vresp,flucts) = get_flucts(shotnum,t0,t1,sensors,method,ignore,order,bdplot)

    (bd) = BD(flucts,index)
    t = time[index]

    dt = t[1]-t[0]
    if correlate:
        (pairs) = correlation(dt,group_index,bd,nummodes=modenum,modes = modes,plot=corplot,exclude = exclude)
    elif modes != None:
        pairs = []
        for i in range(len(modes)//2):
            pairs.append([modes[2*i],modes[2*i+1]])
    else:
        print 'No pairs found!  choosing first set of 2 pairs!'
        pairs = [[0,1],[2,3]]

    if ( (correlate and (pairs == [])) ):
        print 'No pairs found!  choosing first set of 2 pairs!'
        pairs = [[0,1],[2,3]]
    if bdplot:
        fig = plt.figure(12)
        plt.figtext(.35,.95,
                     "Plasma parameters & first {0:d} mode pairs.  Shot #{1:d}".format(len(pairs),shotnum),
                     fontsize = 20)
        for i in range(len(pairs)):
            ax = fig.add_subplot(len(pairs),3,2+3*i)
            plt.title('PA2 Poloidal Pair {0:d}'.format(i+1))
            plt.plot(group['PA2P'],bd[0][group_index['PA2P'],pairs[i][0]],'b')
            plt.plot(group['PA2P'],bd[0][group_index['PA2P'],pairs[i][1]],'g')
            plt.plot(group['PA2P'],bd[0][group_index['PA2P'],pairs[i][0]],'bo')
            plt.plot(group['PA2P'],bd[0][group_index['PA2P'],pairs[i][1]],'go')
            plt.plot(group['PA2P'],bd[0][group_index['PA2P'],pairs[i][0]]*0,'--k')

            if (np.mean(MR[index]) > .93) and (exclude != 'odd'):
                n_array = 'FBP3'
            else:
                n_array = 'TAP'
            ax = fig.add_subplot(len(pairs),3,3+3*i)
            plt.title('{0} Poloidal Pair {1:d}'.format(n_array[:2],(i+1)))
            plt.plot(group[n_array],bd[0][group_index[n_array],pairs[i][0]],'b')
            plt.plot(group[n_array],bd[0][group_index[n_array],pairs[i][1]],'g')
            plt.plot(group[n_array],bd[0][group_index[n_array],pairs[i][0]],'bo')
            plt.plot(group[n_array],bd[0][group_index[n_array],pairs[i][1]],'go')
            plt.plot(group[n_array],bd[0][group_index[n_array],pairs[i][0]]*0,'--k')
            
            fig.subplots_adjust(left = .040, right = .99, top = .90, bottom = .075,hspace = .15,wspace = .15)

        fig = plt.figure(9,figsize = (11,11))
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        plt.plot(np.arange(1,21),100*(bd[1][:20]**2)/(np.sum(bd[1]**2)),'o')
        plt.title('Formula: 100*(sing_val**2/sum(sing_vals**2))')
        plt.ylabel('Percent power in each mode')
        plt.ylim(1e-2,1e2)
        for i in range(len(pairs)):
            plt.plot(pairs[i][0]+1,100*(bd[1][pairs[i][0]]**2)/(np.sum(bd[1]**2)),'m^')        
            plt.plot(pairs[i][1]+1,100*(bd[1][pairs[i][1]]**2)/(np.sum(bd[1]**2)),'yv')

        fig = plt.figure(10)
        ax = fig.add_subplot(111, polar=True)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if exclude == 'odd':
            PA = 'PA2P'
        else:
            PA = 'PA1P'
        plt.plot(np.arange(0,2*np.pi+.01,.01),
                 np.ones(len(np.arange(0,2*np.pi+.01,.01))),'k--')
        plt.plot(np.asarray(group[PA])/32.*2*np.pi-np.pi,
                 bd[0][group_index[PA],pairs[0][0]]+1,'ob')
        plt.plot(np.asarray(group[PA])/32.*2*np.pi-np.pi,
                 bd[0][group_index[PA],pairs[0][1]]+1,'og')
        plt.plot(np.asarray(group[PA])/32.*2*np.pi-np.pi,
                 bd[0][group_index[PA],pairs[0][0]]+1,'b')
        plt.plot(np.asarray(group[PA])/32.*2*np.pi-np.pi,
                 bd[0][group_index[PA],pairs[0][1]]+1,'g')

    return(time,index, group,group_index, Vresp,flucts,bd)

def spatial_correlate(group,group_index,bd):
    bestmatch = np.zeros((6,3))
    for i in range(1,7):
        for p in range(36):
            k=p//6
            j=p%6
            if k!=j:
        #PA2P is missing sensor 14.  Need to account for this.
            #wavelength = (4*i+1e-6)/32.
            #frequency = 32/(4*i+1e-6)
            #print '\n','Frequency:',frequency
            #plt.figure()
                mode1 = bd[0][group_index['PA2P'],j]
                mode2 = bd[0][group_index['PA2P'],k]
                sensors = group['PA2P']
                
                left_shift = mode1[i:]
                right_shift = mode2[:len(mode2)-i]
                left_sens = np.asarray(sensors[i:])
                right_sens = np.asarray(sensors[:len(sensors)-i])
            #plt.plot(left_sens,left_shift,'o')
            #plt.plot(left_sens,left_shift,'b')
            #plt.plot(right_sens+i,right_shift,'o')
            #plt.plot(right_sens+i,right_shift,'g')
            #plt.ylim(-.25,.25)
            #plt.twinx()
            #plt.ylim(-.1,.1)
            #plt.plot([0,35],[0,0])
            #plt.plot([12,12],[-.1,.1])
        #dropouts:
        #bad indices - left shift[12+i], right shift[13-i]

        #right_prep = right_shift[:]
                correlation =(abs(right_shift*left_shift)/
                              np.sum(left_shift*left_shift))

            #print 'bad correlation: ',np.sum(correlation)
            
                new_correlation = np.append(correlation[:(13-i)],
                                            correlation[(14-i):(-19+i)])
                new_correlation = np.append(new_correlation,
                                            correlation[(-18+i):])
                new_lsens = np.append(left_sens[:(13-i)],left_sens[(14-i):(-19+i)])
                new_lsens = np.append(new_lsens,left_sens[(-18+i):])
                print sum(new_correlation),bestmatch[k,1]
                if sum(new_correlation) > bestmatch[k,1]:
                    bestmatch[k,2] = i
                    bestmatch[k,1] = sum(new_correlation)
                    bestmatch[k,0] = j
    return(bestmatch)
            #    print 'better correlation: ',np.sum(new_correlation),'\n'
            #plt.plot(left_sens,correlation,'or')


def RMP_coupling(shotnum,time,ignore = None):
    degree = np.pi/180
    tree = MDSplus.Tree('hbtep2',shotnum)
    
    cc_data = miscmod.cc_info()
    high_no_clip_signal = 3 #we want a large current, but not so large the signal clips
    high_no_clip_sensor = []
    for key,value in cc_data.iteritems():
        if key[8:] =='big':
            name = 'sensors.cc_currents:'+key[:7]
        else:
            continue
            #print name,int(key[2:4])%2
        if (ignore == 'odd') * (int(key[2:4])%2 ==1): 
            continue
        #print name
        (t,I) = Pull_data.pull_data(tree,name,zero = True, duration = [-.001,.015])
        closeness_to_peak_at_three = abs(max(abs(I))-3)
        if high_no_clip_signal > closeness_to_peak_at_three:
            high_no_clip_signal = closeness_to_peak_at_three
            high_no_clip_sensor = name
            print high_no_clip_signal

    (t,I) = Pull_data.pull_data(tree,high_no_clip_sensor,zero = True, 
                                duration = [-.001,.015])
    return(high_no_clip_sensor,t,I)
        

if __name__ == '__main__':
    options = parse_args()
    init_logging(options.quiet, options.debug)    
    corr = not (options.nocorrelate)
    main(options.shotnum,options.start_time*1e-3,options.end_time*1e-3,
         modes=options.modes,modenum=options.modenum,ignore=options.ignore,
         exclude=options.exclude,method=options.method,corplot=options.corplot,
         correlate=corr)
    if not plt.isinteractive():
        plt.show()
