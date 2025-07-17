class MCTOTHelper:
    table = {}
    
    def __init__(self, calibFilename):
        f = open(calibFilename)
        for line in f:
            s = line.split(",")
            if s[0] == "layer":
                continue
            layer = int(s[0])
            strip = int(s[1])
            end = int(s[2])
            adc_fit_k = float(s[3])
            adc_fit_m = float(s[4])
            tot_fit_k = float(s[5])
            tot_fit_m = float(s[6])
            self.table[(layer,strip,end)] = [adc_fit_k,adc_fit_m,tot_fit_k,tot_fit_m]

    
    def get_Tc_index(self,row, end):
        Tc_index = -1
        for i in range(0,8):
            if bool(row["tot_comp_"+str(i)+"_end"+str(end)]):
                Tc_index = i
        return Tc_index
    
    def get_TOT(self,row, end):
        i = self.get_Tc_index(row, end)
            
        return int(row["tot_"+str(i)+"_end"+str(end)])

    def correct_TOT(self, layer, strip, end, rawtot):
        factors = self.table[(layer,strip,end)]
        return (rawtot-factors[3])/factors[2]*factors[0]+factors[1]
