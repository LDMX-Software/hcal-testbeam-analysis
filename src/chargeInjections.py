import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
import copy
import os

def calibrationCurve(dpm, ilink, channel, filename):

    #Choose input file

    #filename = "scan_DPM0_CALIBRUN_coff14_20220424_220633.csv"
    print(os.listdir("../chargeinjections"))

    file = open(filename,"r")

    #Choose a channel

    #dpm = 0
    #ilink = 2
    #channel = 32

    #TODO Translate to layer-bar-end indexing (using map.csv)

    #Prepare plot canvases
    fig, axs = plt.subplots(1,2, figsize=(16*3/4,9*3/4))
    inset = axs[0].inset_axes([0.55,0.05,0.45,0.45])

    #Load data

    class event:
        calib_dac = -1
        capacitor_type = -1
        maxADC = -1
        sumADC = -1
        TOT = -1
        charge = -1

    events = []

    for line in file:
        s = line.split(",")

        #Skip first line
        if(s[0] == "CALIB_DAC"):
            continue

        CALIB_DAC = int(s[0])
        DPM = int(s[1])
        ILINK = int(s[2])
        CHAN = int(s[3])

        if DPM != dpm or ILINK != ilink or CHAN != channel:
            continue

        EVENT = int(s[4])
        ADC0 = int(s[5])
        ADC1 = int(s[6])
        ADC2 = int(s[7])
        ADC3 = int(s[8])
        ADC4 = int(s[9])
        ADC5 = int(s[10])
        ADC6 = int(s[11])
        ADC7 = int(s[12])
        TOT0 = int(s[13])
        TOT1 = int(s[14])
        TOT2 = int(s[15])
        TOT3 = int(s[16])
        TOT4 = int(s[17])
        TOT5 = int(s[18])
        TOT6 = int(s[19])
        TOT7 = int(s[20])
        TOA0 = int(s[21])
        TOA1 = int(s[22])
        TOA2 = int(s[23])
        TOA3 = int(s[24])
        TOA4 = int(s[25])
        TOA5 = int(s[26])
        TOA6 = int(s[27])
        TOA7 = int(s[28])
        CAPACITOR_TYPE = s[29]
        SIPM_BIAS = int(s[30])

        sumADC = ADC0+ADC1+ADC2+ADC3+ADC4+ADC5+ADC6+ADC7
        maxADC = max(ADC0,ADC1,ADC2,ADC3,ADC4,ADC5,ADC6,ADC7) 
        TOT = TOA0+TOT1+TOT2+TOT3+TOT4+TOT5+TOT6+TOT7
        
        e = event()
        e.calib_dac = CALIB_DAC
        e.sumADC = sumADC
        e.maxADC = maxADC
        e.TOT = TOT
        e.capacitor_type = CAPACITOR_TYPE

        events.append(e)

    """
    Clean up data and remove outlier points
    """

    loweventsunfiltered = [x for x in events if x.capacitor_type == "0"]
    higheventsunfiltered = [x for x in events if x.capacitor_type == "1"]

    lowevents = []
    highevents = []

    for cd in list(set([x.calib_dac for x in loweventsunfiltered])):
        subevents = [x for x in loweventsunfiltered if x.calib_dac == cd]

        sumADCs = [x.sumADC for x in subevents]
        TOTs = [x.TOT for x in subevents]
        maxTOT = max(TOTs)

        maxdistance = max(abs(sumADCs[0]-sumADCs[1]), abs(sumADCs[0]-sumADCs[2]), abs(sumADCs[1]-sumADCs[2]))
        maxTOTdistance = max(abs(TOTs[0]-TOTs[1]), abs(TOTs[0]-TOTs[2]), abs(TOTs[1]-TOTs[2]))

        #If they are too far apart, throw away the event
        if (maxdistance < 40 and not maxTOT > 0) or (maxTOT > 0 and maxTOTdistance < 40):
            #Average the three events
            e = event()
            e.calib_dac = cd
            e.sumADC = sum([x.sumADC for x in subevents])/3
            e.maxADC = sum([x.maxADC for x in subevents])/3
            e.TOT = sum([x.TOT for x in subevents])/3
            e.capacitor_type = subevents[0].capacitor_type

            lowevents.append(e)

    for cd in list(set([x.calib_dac for x in higheventsunfiltered])):
        subevents = [x for x in higheventsunfiltered if x.calib_dac == cd]

        sumADCs = [x.sumADC for x in subevents]
        TOTs = [x.TOT for x in subevents]
        maxTOT = max(TOTs)

        maxdistance = max(abs(sumADCs[0]-sumADCs[1]), abs(sumADCs[0]-sumADCs[2]), abs(sumADCs[1]-sumADCs[2]))
        maxTOTdistance = max(abs(TOTs[0]-TOTs[1]), abs(TOTs[0]-TOTs[2]), abs(TOTs[1]-TOTs[2]))

        #If they are too far apart, throw away the event
        if (maxdistance < 40 and not maxTOT > 0) or (maxTOT > 0 and maxTOTdistance < 20):
            #Average the three events
            e = event()
            e.calib_dac = cd
            e.sumADC = sum([x.sumADC for x in subevents])/3
            e.maxADC = sum([x.maxADC for x in subevents])/3
            e.TOT = sum([x.TOT for x in subevents])/3
            e.capacitor_type = subevents[0].capacitor_type

            highevents.append(e)

    """
    Charge conversion
    """

    #First do naive charge conversion CALIB_DAC <--> injected charge
    for e in lowevents:
        e.charge = e.calib_dac/2048*500

    for e in loweventsunfiltered:
        e.charge = e.calib_dac/2048*500

    for e in highevents:
        e.charge = e.calib_dac/2048*8000

    for e in higheventsunfiltered:
        e.charge = e.calib_dac/2048*8000


    #Make a fit to the low capacitor sumADC values
    linearfitcoeffs = np.polyfit([x.charge for x in lowevents], [x.sumADC for x in lowevents], 1)

    print("Linear fit to low. cap charge injections:")
    print(str(linearfitcoeffs[0])+"*x + " +str(linearfitcoeffs[1]))

    #Find a constant correction to the charge, based on the first few high cap. events

    leadinghighevents = [x for x in highevents if x.calib_dac < 100]
    leadinghigheventsuncorrected = copy.deepcopy(leadinghighevents)
    chargecorrectionfit = np.polyfit([x.charge for x in leadinghighevents], [x.sumADC for x in leadinghighevents], 1)

    a = chargecorrectionfit[0]
    b = chargecorrectionfit[1]
    c = linearfitcoeffs[0]
    d = linearfitcoeffs[1]

    for e in highevents:
        e.charge = a/c*e.charge+(b-d)/c

    for e in higheventsunfiltered:
        e.charge = a/c*e.charge+(b-d)/c

    """
    Determine the TOT threshold (in units of charge)
    """
    TOTthreshold = min([x.charge for x in highevents if x.TOT > 0])

    """
    Make fits and calibration curves
    """

    #Linear fit to large TOT values
    def linear_asymptote(x,a,b):
        return a*x+b

    xval = [x.charge for x in highevents if x.charge >= TOTthreshold]
    yval = [x.TOT for x in highevents if x.charge >= TOTthreshold]

    xvalend = [x.charge for x in highevents if x.charge >= TOTthreshold][-5:]
    yvalend = [x.TOT for x in highevents if x.charge >= TOTthreshold][-5:]

    popt, pcov = curve_fit(linear_asymptote, xvalend, yvalend)
    x = np.linspace(min(xval),max(xval),100)
    y = [linear_asymptote(p,*popt) for p in x]
    axs[1].scatter(xval,yval)
    axs[1].plot(x,y, c="green",linestyle="-", label="Asymptote")

    #Remove outlier points, and then improve fit

    highevents = [x for x in highevents if abs(linear_asymptote(x.charge,*popt)-x.TOT) < 500 or x.TOT < TOTthreshold]
    xval = [x.charge for x in highevents if x.charge >= TOTthreshold]
    yval = [x.TOT for x in highevents if x.charge >= TOTthreshold]
    axs[1].scatter(xval,yval,marker="x", label="High Cap. Injections", color="black")

    #The final TOT fitting function
    def func(x,a,b):
        return popt[0]*x+popt[1]-a/(x-b)

    poptpower, pcovpower = curve_fit(func, xval, yval, bounds=([-10000,0],[10000,TOTthreshold-10]))
    x = np.linspace(min(xval),max(xval),100)
    y = [func(p,*poptpower) for p in x]
    #print(poptpower)
    axs[1].plot(x,y, c="red",linestyle="--", label="Fit")

    print("popt")
    print(popt)
    print("Poptpower")
    print(poptpower)

    print(str(popt[0])+"*x + " + str(popt[1])+" - " + str(poptpower[0])+"/(x-"+str(poptpower[1])+")")

    """
    Correct sumADC for TOT events finally!
    """

    #Make a correction table to go from TOT to sumADC
    charges = np.linspace(TOTthreshold,8*2047,1000)

    TOTs = [func(p,*poptpower) for p in charges]
    sumADCs = [linearfitcoeffs[0]*p + linearfitcoeffs[1] for p in charges]

    #This is the TOT threshold in units of TOT
    #threshold = func(TOTthreshold, *poptpower)

    threshold = min([x.TOT for x in highevents if x.charge >= TOTthreshold])

    print("ADC Sum line")
    print(linearfitcoeffs)

    print("Fit parameters:")
    print(popt)
    print(poptpower)

    print("Charge threshold" + str(TOTthreshold))
    print("Threshold: " + str(threshold))

    for p in highevents:
        if p.TOT >= threshold:
            t = p.TOT
            if t > max(TOTs):
                p = max(TOTs)
            index = 0
            for i in range(0,len(TOTs)):
                if TOTs[i] >= t:
                    index = i
                    break

            p.sumADC = sumADCs[i]


    """
    Plotting
    """
    x = np.linspace(min([x.charge for x in lowevents]), max([x.charge for x in highevents]), 2)
    y = [linearfitcoeffs[0]*p + linearfitcoeffs[1] for p in x]
    axs[0].plot(x,y, label="Fit to low cap. region", c="black", linestyle="--")

    inset.plot(x,y, label="Fit to low cap. region", c="black", linestyle="--")

    #Plot low cap. events before filtering
    #axs[0].scatter([x.charge for x in loweventsunfiltered], [x.sumADC for x in loweventsunfiltered], label="Low ADC (Unfiltered)")

    axs[0].scatter([x.charge for x in lowevents], [x.sumADC for x in lowevents], label="Low Cap. Injections", color="blue")
    inset.scatter([x.charge for x in lowevents], [x.sumADC for x in lowevents], label="Low ADC",  color="blue")

    #Plot unfiltered high cap. events
    #axs[0].scatter([x.charge for x in higheventsunfiltered if x.charge < TOTthreshold], [x.sumADC for x in higheventsunfiltered if x.charge < TOTthreshold], label="High ADC (Unfiltered)")

    #Plot high cap. events below TOT threshold
    axs[0].scatter([x.charge for x in highevents if x.charge < TOTthreshold], [x.sumADC for x in highevents if x.charge < TOTthreshold], label="High Cap. Injections", color="green")
    inset.scatter([x.charge for x in highevents if x.charge < TOTthreshold], [x.sumADC for x in highevents if x.charge < TOTthreshold], label="High ADC", color="green")

    #Plot high cap. above TOT threshold
    axs[0].scatter([x.charge for x in highevents if x.charge >= TOTthreshold], [x.sumADC for x in highevents if x.charge >= TOTthreshold], label="High Cap. Injections (using TOT)", marker="x")
    inset.scatter([x.charge for x in highevents if x.charge >= TOTthreshold], [x.sumADC for x in highevents if x.charge >= TOTthreshold], label="High Cap. Injections (using TOT)", marker="x")

    #Plot high capacitor events before linear correction
    #axs[0].scatter([x.charge for x in leadinghigheventsuncorrected], [x.sumADC for x in leadinghigheventsuncorrected], label="High ADC (before correction)", marker="x", c="r")

    #Plot unfiltered TOT
    #plt.scatter([x.charge for x in higheventsunfiltered if x.charge > TOTthreshold], [x.TOT for x in higheventsunfiltered if x.charge > TOTthreshold], label="High TOT (Unfiltered)")

    #Draw TOT Threshold
    axs[0].vlines(TOTthreshold, ymin=0, ymax=8000, label="TOT Threshold", color="red", linestyle="-")
    inset.vlines(TOTthreshold, ymin=0, ymax=8000, label="TOT Threshold", color="red", linestyle="-")


    axs[0].legend()
    axs[0].set_xlabel("Charge [fC]")
    axs[0].set_ylabel("Amplitude [Sum of ADC]")

    axs[1].legend()
    axs[1].set_xlabel("Charge [fC]")
    axs[1].set_ylabel("Time Over Threshold")

    inset.set_xlim((min([x.charge for x in lowevents]),TOTthreshold+500))
    inset.set_ylim((min([x.sumADC for x in lowevents]),max([x.sumADC for x in highevents if x.charge < TOTthreshold])+800))

    plt.show()


    plt.scatter([x.charge for x in lowevents], [x.sumADC for x in lowevents], label="Low Cap. Injections", color="blue")
    plt.scatter([x.charge for x in leadinghigheventsuncorrected], [x.sumADC for x in leadinghigheventsuncorrected], label="High ADC (before correction)", marker="x", c="r")
    plt.plot(x,y, label="Fit to low cap. region", c="black", linestyle="--")
    plt.scatter([x.charge for x in highevents if x.charge < TOTthreshold], [x.sumADC for x in highevents if x.charge < TOTthreshold], label="High Cap. Injections", color="green")
    plt.xlabel("Charge [fC]")
    plt.ylabel("Amplitude [Sum of ADC]")
    plt.legend()
    plt.show()


    plt.scatter([x.charge for x in higheventsunfiltered if x.charge >= TOTthreshold], [x.TOT for x in higheventsunfiltered if x.charge >= TOTthreshold], label="High Cap. Injections", color="green")
    plt.xlabel("Charge [fC]")
    plt.ylabel("TOT")
    plt.show()
    #TODO Write/print results

    print(str(popt[0])+"*x + " + str(popt[1])+" - " + str(poptpower[0])+"/(x-"+str(poptpower[1])+")")

    curve = lambda x: popt[0]*x+popt[1]-poptpower[0]/(x-poptpower[1])
    print(str(linearfitcoeffs[0])+"*x + " +str(linearfitcoeffs[1]))
    linearfit = lambda x: linearfitcoeffs[0]*x+linearfitcoeffs[1]

    return TOTthreshold, curve(TOTthreshold), curve, linearfit
