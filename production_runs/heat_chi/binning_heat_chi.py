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
     average,error_naive=jack_knife_2d(data**2,data,function2)
     #average,error_naive=jack_knife_1d(data,np.var)

     f = open('binning_'+magnitude+'.dat', "w")
     f.write("Final set of parametters for "+magnitude)
     print('naive average',average)
     f.write("\nmean naive ="+str(average))
     f.write("\nsigma naive ="+str(error_naive))


     #Initializing the vars
     bins=data.copy()
     vec_m=np.empty([0])
     vec_s2=np.empty([0])
     vec_aver = np.empty([0])
     f_bins = open('bins_m'+magnitude+'.dat',"w")
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
         aver,s2 = jack_knife_2d(bins**2,bins,function2)
         f_bins.write("\n"+str(m)+"   "+str(aver)+"   "+str(s2))
         vec_aver = np.append(vec_aver,aver)
         vec_s2=np.append(vec_s2,s2)
         bins=(bins[0::2]+bins[1::2])/np.float64(2)
     f_bins.close()
     fig=plt.figure(10,(10,8))
     plt.xscale('log')
     plt.title('Binning results')
     plt.xlabel('m')
     plt.ylabel('$\sigma_m$')
     plt.plot(vec_m,vec_s2,'x',color='black',markersize=5,label='data')
     #plt.plot(vec_m,np.exp(y_fit)/N,'-.',color='black',linewidth=0.5, label='fit')
     plt.legend(loc='best')
     plt.show()
     fig.savefig('binning_'+magnitude+'.png')

     average1 = np.sum(vec_aver)/(np.float64(len(vec_aver)))

     tf = 0.5*np.max(vec_s2)**2/vec_s2[-1]**2 -1
     af = np.max(vec_s2)
     corr_estim = vec_s2[-1]**2/vec_s2[0]**2
     f.write("\n<E>/N binning ="+str(average1))
     f.write("\nVar(A)="+str(af**2)+"\tSigma for <A>="+str(af))
     f.write("\ntau exp fit="+str(tf))
     f.write("\nCorrelation time estimated="+str(corr_estim))
     jk_var , error_var = jack_knife_2d(array**2,array,function2)
     f.write("\njk_var"+str(jk_var)+"   "+str(error_var))
     f.close()



     return average1, af ,  tf

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


average1,sigma1 ,tt1=binning(energy20,'heat_capacity T = 2.0',N,2000)
average2,sigma2 ,tt2=binning(energy21,'heat_capacity T = 2.1',N,5000)
average3,sigma3 ,tt3=binning(energy22,'heat_capacity T = 2.2',N,5000)
average12,sigma12 ,tt11=binning(energy_crit,'heat_capacity T = 2.269',N,1000000)
average4,sigma4,tt4=binning(energy23,'heat_capacity T = 2.3',N,100000)
average5,sigma5,tt5=binning(energy24,'heat_capacity T = 2.4',N,3000)
average6,sigma6,tt6=binning(energy25,'heat_capacity T = 2.5',N,5000)
average7,sigma7,tt7=binning(energy26,'heat_capacity T = 2.6',N,1000)
average8,sigma8,tt8=binning(energy27,'heat_capacity T = 2.7',N,1000)
average9,sigma9,tt9=binning(energy28,'heat_capacity T = 2.8',N,1000)
average10,sigma10 ,tt10=binning(energy29,'heat_capacity T = 2.9',N,1000)
average11,sigma11 ,tt11=binning(energy30,'heat_capacity T = 3.0',N,1000)
#
#f1 = open('heat_binning.dat', "w")
#f1.write("2.0"+"   "+str(average1/(np.float64(N)*2**2))+"   "+str(sigma1/(np.float64(N)*2**2)))
#f1.write("\n2.1"+"   "+str(average2/(np.float64(N)*2.1**2))+"   "+str(sigma2/(np.float64(N)*2.1**2)))
#f1.write("\n2.2"+"   "+str(average3/(np.float64(N)*2.2**2))+"   "+str(sigma3/(np.float64(N)*2.2**2)))
#f1.write("\n2.269"+"   "+str(average12/(np.float64(N)*2.269**2))+"   "+str(sigma12/(np.float64(N)*2.26**2)))
#f1.write("\n2.3"+"   "+str(average4/(np.float64(N)*2.3**2))+"   "+str(sigma4/(np.float64(N)*2.3**2)))
#f1.write("\n2.4"+"   "+str(average5/(np.float64(N)*2.4**2))+"   "+str(sigma5/(np.float64(N)*2.4**2)))
#f1.write("\n2.5"+"   "+str(average6/(np.float64(N)*2.5**2))+"   "+str(sigma6/(np.float64(N)*2.5**2)))
#f1.write("\n2.6"+"   "+str(average7/(np.float64(N)*2.6**2))+"   "+str(sigma7/(np.float64(N)*2.6**2)))
#f1.write("\n2.7"+"   "+str(average8/(np.float64(N)*2.7**2))+"   "+str(sigma8/(np.float64(N)*2.7**2)))
#f1.write("\n2.8"+"   "+str(average9/(np.float64(N)*2.8**2))+"   "+str(sigma9/(np.float64(N)*2.8**2)))
#f1.write("\n2.9"+"   "+str(average10/(np.float64(N)*2.9**2))+"   "+str(sigma10/(np.float64(N)*2.9**2)))
#f1.write("\n3.0"+"   "+str(average11/(np.float64(N)*3.0**2))+"   "+str(sigma11/(np.float64(N)*3.0**2)))
#f1.close()

average1,sigma1 ,tt1=binning(M20,'chi T = 2.0',N,1000)
average2,sigma2 ,tt2=binning(M21,'chi T = 2.1',N,1000)
average3,sigma3 ,tt3=binning(M22,'chi T = 2.2',N,10000)
average12,sigma12 ,tt11=binning(M_crit,'chi T = 2.269',N,1000000)
average4,sigma4,tt4=binning(M23,'chi T = 2.3',N,80000)
average5,sigma5,tt5=binning(M24,'chi T = 2.4',N,100000)
average6,sigma6,tt6=binning(M25,'chi T = 2.5',N,1000)
average7,sigma7,tt7=binning(M26,'chi T = 2.6',N,1000)
average8,sigma8,tt8=binning(M27,'chi T = 2.7',N,1000)
average9,sigma9,tt9=binning(M28,'chi T = 2.8',N,4000)
average10,sigma10 ,tt10=binning(M29,'chi T = 2.9',N,2000)
average11,sigma11 ,tt11=binning(M30,'chi T = 3.0',N,100)

#f2 = open('chi_binning.dat', "w")
#f2.write("2.0"+"   "+str(average1/(np.float64(N)*2))+"   "+str(sigma1/np.float64(N)*2))
#f2.write("\n2.1"+"   "+str(average2/(np.float64(N)*2.1))+"   "+str(sigma2/(np.float64(N)*2.1)))
#f2.write("\n2.2"+"   "+str(average3/(np.float64(N)*2.2))+"   "+str(sigma3/(np.float64(N)*2.2)))
#f2.write("\n2.269"+"   "+str(average12/(np.float64(N)*2.26))+"   "+str(sigma12/(np.float64(N)*2.26)))
#f2.write("\n2.3"+"   "+str(average4/(np.float64(N)*2.3))+"   "+str(sigma4/(np.float64(N)*2.3)))
#f2.write("\n2.4"+"   "+str(average5/(np.float64(N)*2.4))+"   "+str(sigma5/(np.float64(N)*2.4)))
#f2.write("\n2.5"+"   "+str(average6/(np.float64(N)*2.5))+"   "+str(sigma6/(np.float64(N)*2.5)))
#f2.write("\n2.6"+"   "+str(average7/(np.float64(N)*2.6))+"   "+str(sigma7/(np.float64(N)*2.6)))
#f2.write("\n2.7"+"   "+str(average8/(np.float64(N)*2.7))+"   "+str(sigma8/(np.float64(N)*2.7)))
#f2.write("\n2.8"+"   "+str(average9/(np.float64(N)*2.8))+"   "+str(sigma9/(np.float64(N)*2.8)))
#f2.write("\n2.9"+"   "+str(average10/(np.float64(N)*2.9))+"   "+str(sigma10/(np.float64(N)*2.9)))
#f2.write("\n3.0"+"   "+str(average11/(np.float64(N)*3.0))+"   "+str(sigma11/(np.float64(N)*3.0)))
#f2.close()
