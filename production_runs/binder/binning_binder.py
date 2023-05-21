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
     average,error_naive=jack_knife_2d(data**4,data**2,function3)
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
         aver,s2 = jack_knife_2d(bins**4,bins**2,function3)
         vec_aver = np.append(vec_aver,aver)
         #s2=np.sum((bins-aver)**2.0)/np.float64((N_b)*(N_b-1))
         #s2 = np.sqrt(s2)

         vec_s2=np.append(vec_s2,s2)

         bins=(bins[0::2]+bins[1::2])/np.float64(2)


     #Getting the parameters for the fitting
     try:
          #trying to fit the curve
          popt,pcov=curve_fit(fitting,vec_m,np.log(vec_s2))
          af,bf,tf=popt
          y_fit=fitting(vec_m,af,bf,tf)
          #Ploting the results
          plt.style.use(['science'])
          fig=plt.figure(10,(10,8))
          plt.xscale('log')
          plt.title('Binning results')
          plt.xlabel('m')
          plt.ylabel('$\sigma_m$')
          plt.plot(vec_m,vec_s2,'x',color='black',markersize=5,label='data')
          plt.plot(vec_m,np.exp(y_fit),'-.',color='black',linewidth=0.5, label='fit')
          plt.legend(loc='best')
          plt.show()
          fig.savefig('binning_S2_'+magnitude+'.png')
     except:
          #if the fit fails we get the avg of the binning
          af=np.max(vec_s2)
          print('failed')
          tf=np.max(vec_s2)**2/vec_s2[-1]**2
     
     average1 = np.sum(vec_aver)/(np.float64(len(vec_aver)))
     if(vec_s2[-1]<np.max(vec_s2)):
         corr_max = np.max(vec_s2)**2/vec_s2[-1]**2
         af_max = np.max(vec_s2)
         f.write("\nsigma_max="+str(af_max)+"\ttau_max="+str(corr_max))

     corr_estim = vec_s2[-1]**2/vec_s2[0]**2
     f.write("\n<E>/N binning ="+str(average1))
     f.write("\nVar(A)="+str(af**2)+"\tSigma for <A>="+str(af))
     f.write("\ntau exp fit="+str(tf))
     f.write("\nCorrelation time estimated="+str(corr_estim))
     jk_var , error_var = jack_knife_2d(array**4,array**2,function3)
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

def function3(x,y):
    return 0.5*(3-x/y**2)

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

M1 = test1[:,2]
M2 = test2[:,2]
M3 = test3[:,2]
M4 = test4[:,2]
M5 =  test5[:,2]
M6 =  test6[:,2]
M7 =  test7[:,2]
M8 =  test8[:,2]
M9 =  test9[:,2]
M10 =  test10[:,2]
M11 =  test11[:,2]
M12 =  test12[:,2]



steps1 = steps[2000:]

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




average1,sigma1 ,tt1=binning(M20,'binder T = 2.0',N,10000)
average2,sigma2 ,tt2=binning(M21,'binder T = 2.1',N,10000)
average3,sigma3 ,tt3=binning(M22,'binder T = 2.2',N,10000000)
average12,sigma12 ,tt11=binning(M_crit,'chi T = 2.269',N,10000000)
average4,sigma4,tt4=binning(M23,'binder T = 2.3',N,10000000)
average5,sigma5,tt5=binning(M24,'binder T = 2.4',N,10000)
average6,sigma6,tt6=binning(M25,'binder T = 2.5',N,10000)
average7,sigma7,tt7=binning(M26,'binder T = 2.6',N,10000)
average8,sigma8,tt8=binning(M27,'binder T = 2.7',N,10000)
average9,sigma9,tt9=binning(M28,'binder T = 2.8',N,10000)
average10,sigma10 ,tt10=binning(M29,'binder T = 2.9',N,10000)
average11,sigma11 ,tt11=binning(M30,'binder T = 3.0',N,10000)

f2 = open('binder_binning.dat', "w")
f2.write("2.0"+"   "+str(average1)+"   "+str(sigma1))
f2.write("\n2.1"+"   "+str(average2)+"   "+str(sigma2))
f2.write("\n2.2"+"   "+str(average3)+"   "+str(sigma3))
f2.write("\n2.269"+"   "+str(average12)+"   "+str(sigma12))
f2.write("\n2.3"+"   "+str(average4)+"   "+str(sigma4))
f2.write("\n2.4"+"   "+str(average5)+"   "+str(sigma5))
f2.write("\n2.5"+"   "+str(average6)+"   "+str(sigma6))
f2.write("\n2.6"+"   "+str(average7)+"   "+str(sigma7))
f2.write("\n2.7"+"   "+str(average8)+"   "+str(sigma8))
f2.write("\n2.8"+"   "+str(average9)+"   "+str(sigma9))
f2.write("\n2.9"+"   "+str(average10)+"   "+str(sigma10))
f2.write("\n3.0"+"   "+str(average11)+"   "+str(sigma11))
f2.close()
