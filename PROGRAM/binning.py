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

test1 = np.loadtxt('results.out')


with open('input.dat') as f:
    first_line = f.readline().strip()
L = int(first_line[0]+first_line[1]+first_line[2])
print('ewa',L)
N = L*L
f.close()

steps = test1[:,0]

energy1 = test1[:,1]
M1 = np.abs(test1[:,2])
steps1 = steps[2000:]
energy20 = energy1[2000:]

M20 = M1[2000:]


average1,sigma1,b1 ,tt1=binning(energy20,'Energy T = 2.0',N,10000)

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
