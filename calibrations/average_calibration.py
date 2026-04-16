f = open("mc_v8_tot_calibration.csv","r")

fout = open("mc_v8_tot_calibration_averaged.csv","w")

fout.write("layer,strip,end,adc_fit_k,adc_fit_m,tot_fit_k,tot_fit_m\n")


adc_fit_k = []
adc_fit_m = []
tot_fit_k = []
tot_fit_m = []

for line in f:
    s = line.split(",")
    if s[0] == "layer":
        continue
    print(s[0])


    adc_fit_k.append(float(s[3]))
    adc_fit_m.append(float(s[4]))
    tot_fit_k.append(float(s[5]))
    tot_fit_m.append(float(s[6]))

adc_fit_k_avg = sum(adc_fit_k)/len(adc_fit_k)
adc_fit_m_avg = sum(adc_fit_m)/len(adc_fit_m)
tot_fit_k_avg = sum(tot_fit_k)/len(tot_fit_k)
tot_fit_m_avg = sum(tot_fit_k)/len(tot_fit_m)

for layer in range(1,20):
    for strip in range(0,13):
        for end in [0,1]:
            fout.write(''.join([str(x)+"," for x in [layer,strip,end,adc_fit_k_avg,adc_fit_m_avg,tot_fit_k_avg,tot_fit_m_avg]])[:-1]+"\n")
