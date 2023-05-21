# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Tue Jun 23 01:55:33 2020

@author: asure
"""
import numpy as np
import matplotlib.pyplot as plt


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

tau20 = 4
tau21 = 6
tau22 = 14
tau23 = 20
tau24 = 10
tau25 = 6
tau26 = 4
tau27 = 3
tau28 = 3
tau29 = 2
tau30 = 2
taucrit = 20 
steps1 = steps[2000:]

energy20 = energy1[2000::tau20]
energy21 = energy2[2000::tau21]
energy22 = energy3[2000::tau22]
energy23 = energy4[2000::tau23]
energy24 = energy5[2000::tau24]
energy25 = energy6[2000::tau25]
energy26 = energy7[2000::tau26]
energy27 = energy8[2000::tau27]
energy28 = energy9[2000::tau28]
energy29 = energy10[2000::tau29]
energy30 = energy11[2000::tau30]
energy_crit = energy12[2000::taucrit]

tau_20 = 5
tau_21 = 8
tau_22 = 28
tau_23 = 36
tau_24 = 26
tau_25 = 8
tau_26 = 7
tau_27 = 3
tau_28 = 3
tau_29 = 2
tau_30 = 2
tau_crit = 36 

M20 = M1[2000::tau_20]
M21 = M2[2000::tau_21]
M22 = M3[2000::tau_22]
M23 = M4[2000::tau_23]
M24 = M5[2000::tau_24]
M25 = M6[2000::tau_25]
M26 = M7[2000::tau_26]
M27 = M8[2000::tau_27]
M28 = M9[2000::tau_28]
M29 = M10[2000::tau_29]
M30 = M11[2000::tau_30]
M_crit = M12[2000::tau_crit]


average1,sigma1=jack_knife_2d(energy20**2,energy20,function2)
average2,sigma2=jack_knife_2d(energy21**2,energy21,function2)
average3,sigma3=jack_knife_2d(energy22**2,energy22,function2)
average12,sigma12=jack_knife_2d(energy_crit**2,energy_crit,function2)
average4,sigma4=jack_knife_2d(energy23**2,energy23,function2)
average5,sigma5=jack_knife_2d(energy24**2,energy24,function2)
average6,sigma6=jack_knife_2d(energy25**2,energy25,function2)
average7,sigma7=jack_knife_2d(energy26**2,energy26,function2)
average8,sigma8=jack_knife_2d(energy27**2,energy27,function2)
average9,sigma9=jack_knife_2d(energy28**2,energy28,function2)
average10,sigma10=jack_knife_2d(energy29**2,energy29,function2)
average11,sigma11=jack_knife_2d(energy30**2,energy30,function2)

f1 = open('heat.dat', "w")
f1.write("2.0"+"   "+str(average1/(np.float64(N)*2**2))+"   "+str(sigma1/(np.float64(N)*2**2)))
f1.write("\n2.1"+"   "+str(average2/(np.float64(N)*2.1**2))+"   "+str(sigma2/(np.float64(N)*2.1**2)))
f1.write("\n2.2"+"   "+str(average3/(np.float64(N)*2.2**2))+"   "+str(sigma3/(np.float64(N)*2.2**2)))
f1.write("\n2.269"+"   "+str(average12/(np.float64(N)*2.269**2))+"   "+str(sigma12/(np.float64(N)*2.269**2)))
f1.write("\n2.3"+"   "+str(average4/(np.float64(N)*2.3**2))+"   "+str(sigma4/(np.float64(N)*2.3**2)))
f1.write("\n2.4"+"   "+str(average5/(np.float64(N)*2.4**2))+"   "+str(sigma5/(np.float64(N)*2.4**2)))
f1.write("\n2.5"+"   "+str(average6/(np.float64(N)*2.5**2))+"   "+str(sigma6/(np.float64(N)*2.5**2)))
f1.write("\n2.6"+"   "+str(average7/(np.float64(N)*2.6**2))+"   "+str(sigma7/(np.float64(N)*2.6**2)))
f1.write("\n2.7"+"   "+str(average8/(np.float64(N)*2.7**2))+"   "+str(sigma8/(np.float64(N)*2.7**2)))
f1.write("\n2.8"+"   "+str(average9/(np.float64(N)*2.8**2))+"   "+str(sigma9/(np.float64(N)*2.8**2)))
f1.write("\n2.9"+"   "+str(average10/(np.float64(N)*2.9**2))+"   "+str(sigma10/(np.float64(N)*2.9**2)))
f1.write("\n3.0"+"   "+str(average11/(np.float64(N)*3.0**2))+"   "+str(sigma11/(np.float64(N)*3.0**2)))
f1.close()

average1,sigma1=jack_knife_2d(M20**2,M20,function2)
average2,sigma2=jack_knife_2d(M21**2,M21,function2)
average3,sigma3=jack_knife_2d(M22**2,M22,function2)
average12,sigma12=jack_knife_2d(M_crit**2,M_crit,function2)
average4,sigma4=jack_knife_2d(M23**2,M23,function2)
average5,sigma5=jack_knife_2d(M24**2,M24,function2)
average6,sigma6=jack_knife_2d(M25**2,M25,function2)
average7,sigma7=jack_knife_2d(M26**2,M26,function2)
average8,sigma8=jack_knife_2d(M27**2,M27,function2)
average9,sigma9=jack_knife_2d(M28**2,M28,function2)
average10,sigma10=jack_knife_2d(M29**2,M29,function2)
average11,sigma11=jack_knife_2d(M30**2,M30,function2)


f2 = open('chi.dat', "w")
f2.write("2.0"+"   "+str(average1/(np.float64(N)*2))+"   "+str(sigma1/np.float64(N)))
f2.write("\n2.1"+"   "+str(average2/(np.float64(N)*2.1))+"   "+str(sigma2/(np.float64(N)*2.1)))
f2.write("\n2.2"+"   "+str(average3/(np.float64(N)*2.2))+"   "+str(sigma3/(np.float64(N)*2.2)))
f2.write("\n2.269"+"   "+str(average12/(np.float64(N)*2.269))+"   "+str(sigma12/(np.float64(N)*2.269)))
f2.write("\n2.3"+"   "+str(average4/(np.float64(N)*2.3))+"   "+str(sigma4/(np.float64(N)*2.3)))
f2.write("\n2.4"+"   "+str(average5/(np.float64(N)*2.4))+"   "+str(sigma5/(np.float64(N)*2.4)))
f2.write("\n2.5"+"   "+str(average6/(np.float64(N)*2.5))+"   "+str(sigma6/(np.float64(N)*2.5)))
f2.write("\n2.6"+"   "+str(average7/(np.float64(N)*2.6))+"   "+str(sigma7/(np.float64(N)*2.6)))
f2.write("\n2.7"+"   "+str(average8/(np.float64(N)*2.7))+"   "+str(sigma8/(np.float64(N)*2.7)))
f2.write("\n2.8"+"   "+str(average9/(np.float64(N)*2.8))+"   "+str(sigma9/(np.float64(N)*2.8)))
f2.write("\n2.9"+"   "+str(average10/(np.float64(N)*2.9))+"   "+str(sigma10/(np.float64(N)*2.9)))
f2.write("\n3.0"+"   "+str(average11/(np.float64(N)*3.0))+"   "+str(sigma11/(np.float64(N)*3.0)))
f2.close()



