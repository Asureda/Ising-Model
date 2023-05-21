import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def binning(array,magnitude,N,m_max):
     '''
     Function that returns the average
     and the standard deviation(w/ binning method)
     for a data array
     array :: NUMPY TYPE
     REURN
        - average
        - variance
        - correlation time

     '''

     def fitting(x,a,b,t):

     #Function to fit the S^2(m) to the the variance and the correlation time
     #   a : variance of the data set
     #   t : correlation time
         return np.log(a-b*np.exp(-x/t))
     # Converting to float64
     data=np.array(array,dtype=np.float64)
     #computug m as we need a power of 2
     x=np.int(np.log2(len(data)))
     #Getting the lower optimal size as a power of two
     n=2**x
     print(n)
     # array for the binnig with the optimal size to perform the algorithm
     data=data[0:n]
     average=np.sum(data)/(np.float64(len(data)))
     f = open('binning_'+magnitude+'.dat', "w")
     f.write("Final set of parametters for "+magnitude)
     print('naive average',average/np.float64(N))
     f.write("\n<X>/N naive ="+str(average/np.float64(N)))
     f_bins = open('bins_m'+magnitude+'.dat',"w")

     #Initializing the vars
     bins=data.copy()
     vec_m=np.empty([0])
     vec_s2=np.empty([0])
     vec_aver = np.empty([0])

     #______________________________BINNING____________________________________
     for i in range(0,x):
         #in every iteration we add two consecutive numbers and get the average.
         #in every iteration we get the average as the sum for the array divided
         #for the number of bins
         #Then e compute the s^2
         m=2**i
         N_b=int(n/m)
         if(m>m_max):
             break
         vec_m=np.append(vec_m,m)
         #aver = np.sum(bins)/np.float64(N_b)
         aver,s2 = jack_knife_1d(bins,function1)
         vec_aver = np.append(vec_aver,aver)
         #s2=np.sum((bins-aver)**2.0)/np.float64((N_b)*(N_b-1))
         #s2 = np.sqrt(s2)
         f_bins.write("\n "+str(aver)+"   "+str(s2))

         vec_s2=np.append(vec_s2,s2)

         bins=(bins[0::2]+bins[1::2])/np.float64(2)
     f_bins.close()


     #Getting the parameters for the fitting
     try:
          #trying to fit the curve
          popt,pcov=curve_fit(fitting,vec_m,np.log(vec_s2))
          af,bf,tf=popt
          y_fit=fitting(vec_m,af,bf,tf)
          #Ploting the results
          fig=plt.figure()
          plt.xscale('log')
          plt.title('Binning results'+ ' '+magnitude)
          plt.xlabel('m')
          plt.ylabel('$\sigma_m$')
          plt.plot(vec_m,vec_s2,'x',color='black',markersize=5,label='data')
          plt.plot(vec_m,np.exp(y_fit),'-.',color='black',linewidth=0.5, label='fit')
          plt.legend(loc='best')
          plt.show()
          fig.savefig('binning_S2_'+magnitude+'.png')
     except:
          #if the fit fails we get the avg of the binning
          af=np.sum(vec_s2)/np.float64(len(vec_s2))

          tf=0.0

     average1 = np.sum(vec_aver)/(np.float64(len(vec_aver)))
     corr_estim = vec_s2[-1]**2/vec_s2[0]**2
     f.write("\n<E>/N binning ="+str(average1/np.float64(N)))
     f.write("\nVar(A)="+str(af**2/np.float64(N))+"\tSigma for <A>="+str(af/np.float64(N)))
     f.write("\ntau exp fit="+str(tf))
     f.write("\ntau int="+str(corr_estim))
     jk_avg,jk_err = jack_knife_1d(array,function1)
     f.write("\njk_avg"+str(jk_avg/N)+"   "+str(jk_err/N))
     f.close()



     return average1, af , bf, tf

def jack_knife_1d(x,function):
    x_total=np.sum(x)
    jk_aver_x=[]
    for i in x:
        jk_aver_x.append((x_total-i)/np.float64(len(x)-1))

    jk_func=np.array(list(map(lambda x : function(x),jk_aver_x)))
    jk_func_aver=np.sum(jk_func)/np.float64(len(x))

    err=np.sqrt((np.float64(len(x)-1))*np.sum((jk_func-jk_func_aver)**2.)/np.float64(len(x)))
    return jk_func_aver,err

def jack_knife_2d(x,y,function):
    x_total=np.sum(x)
    y_total=np.sum(y)
    jk_aver_x=[]
    jk_aver_y=[]
    for i,j in zip(x,y):
      jk_aver_x.append((x_total-i)/np.float64(len(x)-1))
      jk_aver_y.append((y_total-j)/np.float64(len(y)-1))
      
    jk_func=[]
    jk_func=np.array(list(map(lambda x,y : function(x,y),jk_aver_x,jk_aver_y )))
    jk_func_aver=np.sum(jk_func)/np.float64(len(x))
    
    err=np.sqrt((np.float64(len(x)-1))*np.sum((jk_func-jk_func_aver)**2.)/np.float64(len(x)))
    return jk_func_aver,err

def auto_correlation(v):
    # np.correlate computes C_{v}[k] = sum_n v[n+k] * v[n]
    corr = np.correlate(v,v,mode="full") # correlate returns even array [0:2*nums-1] cent
    return corr[len(v)-1:]/len(v) # take positive values and normalize by number of point

def function1(x):
    return x

def function2(x,y):
    return x-y**2

test1 = np.loadtxt('results_20.out')
test2 = np.loadtxt('results_21.out')
test3 = np.loadtxt('results_22.out')
test4 = np.loadtxt('results_23.out')
test5 = np.loadtxt('results_24.out')
test6 = np.loadtxt('results_25.out')
test7 = np.loadtxt('results_26.out')
test8 = np.loadtxt('results_27.out')
test9 = np.loadtxt('results_28.out')
test10 = np.loadtxt('results_29.out')
test11 = np.loadtxt('results_30.out')
test12 = np.loadtxt('results_Tcrit.out')


with open('input.dat') as f:
    first_line = f.readline().strip()
L = int(first_line[0]+first_line[1]+first_line[2])
print('ewa',L)
N = L*L
f.close()

steps = test1[:,0]

energy1 = test1[:,1]
energy2 = test2[:,1]
energy3 = test3[:,1]
energy4 = test4[:,1]
energy5 =  test5[:,1]
energy6 =  test6[:,1]
energy7 =  test7[:,1]
energy8 =  test8[:,1]
energy9 =  test9[:,1]
energy10 =  test10[:,1]
energy11 =  test11[:,1]
energy12 =  test12[:,1]

M1 = np.abs(test1[:,2])
M2 = np.abs(test2[:,2])
M3 =  np.abs(test3[:,2])
M4 =  np.abs(test4[:,2])
M5 =  np.abs(test5[:,2])
M6 =  np.abs(test6[:,2])
M7 =  np.abs(test7[:,2])
M8 =  np.abs(test8[:,2])
M9 =  np.abs(test9[:,2])
M10 =  np.abs(test10[:,2])
M11 =  np.abs(test11[:,2])
M12 =  np.abs(test12[:,2])

steps1 = steps[2000:]
energy20 = energy1[2000:]
energy21 = energy2[2000:]
energy22 = energy3[2000:]
energy23 = energy4[2000:]
energy24 = energy5[2000:]
energy25 = energy6[2000:]
energy26 = energy7[2000:]
energy27 = energy8[2000:]
energy28 = energy9[2000:]
energy29 = energy10[2000:]
energy30 = energy11[2000:]
energy_crit = energy12[2000:]

M20 = M1[2000:]
M21 = M2[2000:]
M22 = M3[2000:]
M23 = M4[2000:]
M24 = M5[2000:]
M25 = M6[2000:]
M26 = M7[2000:]
M27 = M8[2000:]
M28 = M9[2000:]
M29 = M10[2000:]
M30 = M11[2000:]
M_crit = M12[2000:]


average1,sigma1,b1 ,tt1=binning(energy20,'Energy T = 2.0',N,2000)
average2,sigma2,b2 ,tt2=binning(energy21,'Energy T = 2.1',N,5000)
average3,sigma3,b3 ,tt3=binning(energy22,'Energy T = 2.2',N,5000)
average12,sigma12,b12 ,tt11=binning(energy_crit,'Energy T = 2.27',N,200000)
average4,sigma4,b4 ,tt4=binning(energy23,'Energy T = 2.3',N,100000)
average5,sigma5,b5 ,tt5=binning(energy24,'Energy T = 2.4',N,3000)
average6,sigma6,b6 ,tt6=binning(energy25,'Energy T = 2.5',N,5000)
average7,sigma7,b7 ,tt7=binning(energy26,'Energy T = 2.6',N,1000)
average8,sigma8,b8 ,tt8=binning(energy27,'Energy T = 2.7',N,1000)
average9,sigma9,b9 ,tt9=binning(energy28,'Energy T = 2.8',N,1000)
average10,sigma10,b10 ,tt10=binning(energy29,'Energy T = 2.9',N,1000)
average11,sigma11,b11 ,tt11=binning(energy30,'Energy T = 3.0',N,1000)

#f1 = open('ENERGY.dat', "w")
#f1.write("2.0"+"   "+str(average1/np.float64(N))+"   "+str(sigma1/np.float64(N)))
#f1.write("\n2.1"+"   "+str(average2/np.float64(N))+"   "+str(sigma2/np.float64(N)))
#f1.write("\n2.2"+"   "+str(average3/np.float64(N))+"   "+str(sigma3/np.float64(N)))
#f1.write("\n2.27"+"   "+str(average12/np.float64(N))+"   "+str(sigma12/np.float64(N)))
#f1.write("\n2.3"+"   "+str(average4/np.float64(N))+"   "+str(sigma4/np.float64(N)))
#f1.write("\n2.4"+"   "+str(average5/np.float64(N))+"   "+str(sigma5/np.float64(N)))
#f1.write("\n2.5"+"   "+str(average6/np.float64(N))+"   "+str(sigma6/np.float64(N)))
#f1.write("\n2.6"+"   "+str(average7/np.float64(N))+"   "+str(sigma7/np.float64(N)))
#f1.write("\n2.7"+"   "+str(average8/np.float64(N))+"   "+str(sigma8/np.float64(N)))
#f1.write("\n2.8"+"   "+str(average9/np.float64(N))+"   "+str(sigma9/np.float64(N)))
#f1.write("\n2.9"+"   "+str(average10/np.float64(N))+"   "+str(sigma10/np.float64(N)))
#f1.write("\n3.0"+"   "+str(average11/np.float64(N))+"   "+str(sigma11/np.float64(N)))
#f1.close()

average1,sigma1,b1 ,tt1=binning(M20,'Magnetization T = 2.0',N,100000)
average2,sigma2,b2 ,tt2=binning(M21,'Magnetization T = 2.1',N,200000)
average3,sigma3,b3 ,tt3=binning(M22,'Magnetization T = 2.2',N,400000)
average12,sigma12,b12 ,tt11=binning(M_crit,'Magnetization T = 2.27',N,300000)
average4,sigma4,b4 ,tt4=binning(M23,'Magnetization T = 2.3',N,100000)
average5,sigma5,b5 ,tt5=binning(M24,'Magnetization T = 2.4',N,200000)
average6,sigma6,b6 ,tt6=binning(M25,'Magnetization T = 2.5',N,20000)
average7,sigma7,b7 ,tt7=binning(M26,'Magnetization T = 2.6',N,20000)
average8,sigma8,b8 ,tt8=binning(M27,'Magnetization T = 2.7',N,20000)
average9,sigma9,b9 ,tt9=binning(M28,'Magnetization T = 2.8',N,20000)
average10,sigma10,b10 ,tt10=binning(M29,'Magnetization T = 2.9',N,20000)
average11,sigma11,b11 ,tt11=binning(M30,'Magnetization T = 3.0',N,10000)

#f2 = open('MAGNETIZATION.dat', "w")
#f2.write("2.0"+"   "+str(average1/np.float64(N))+"   "+str(sigma1/np.float64(N)))
#f2.write("\n2.1"+"   "+str(average2/np.float64(N))+"   "+str(sigma2/np.float64(N)))
#f2.write("\n2.2"+"   "+str(average3/np.float64(N))+"   "+str(sigma3/np.float64(N)))
#f2.write("\n2.27"+"   "+str(average12/np.float64(N))+"   "+str(sigma12/np.float64(N)))
#f2.write("\n2.3"+"   "+str(average4/np.float64(N))+"   "+str(sigma4/np.float64(N)))
#f2.write("\n2.4"+"   "+str(average5/np.float64(N))+"   "+str(sigma5/np.float64(N)))
#f2.write("\n2.5"+"   "+str(average6/np.float64(N))+"   "+str(sigma6/np.float64(N)))
#f2.write("\n2.6"+"   "+str(average7/np.float64(N))+"   "+str(sigma7/np.float64(N)))
#f2.write("\n2.7"+"   "+str(average8/np.float64(N))+"   "+str(sigma8/np.float64(N)))
#f2.write("\n2.8"+"   "+str(average9/np.float64(N))+"   "+str(sigma9/np.float64(N)))
#f2.write("\n2.9"+"   "+str(average10/np.float64(N))+"   "+str(sigma10/np.float64(N)))
#f2.write("\n3.0"+"   "+str(average11/np.float64(N))+"   "+str(sigma11/np.float64(N)))
#f2.close()


