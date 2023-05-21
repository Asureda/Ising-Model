import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#import binning as st

plt.style.use(['science'])

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
##Getting the lower optimal size as a power of two
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


steps1 = steps[0:1000]
energy20 = energy1[0:1000]
energy21 = energy2[0:1000]
energy22 = energy3[0:1000]
energy23 = energy4[0:1000]
energy24 = energy5[0:1000]
energy25 = energy6[0:1000]
energy26 = energy7[0:1000]
energy27 = energy8[0:1000]
energy28 = energy9[0:1000]
energy29 = energy10[0:1000]
energy30 = energy11[0:1000]
energy_crit = energy12[0:1000]

M20 = M1[0:1000]
M21 = M2[0:1000]
M22 = M3[0:1000]
M23 = M4[0:1000]
M24 = M5[0:1000]
M25 = M6[0:1000]
M26 = M7[0:1000]
M27 = M8[0:1000]
M28 = M9[0:1000]
M29 = M10[0:1000]
M30 = M11[0:1000]
M_crit = M12[0:1000]



fig=plt.figure(12,(12,10))
#

plt.title('Energy time series',fontsize = 16)
plt.xlabel('MCS',fontsize=16)
plt.ylabel(r'$E/N$',fontsize=16)
plt.xlim(0,1000)


plt.plot(steps1, energy20/N,'--',markersize=4,label='T=2.0');
plt.plot(steps1, energy21/N,'--',markersize=4,label='T=2.1');
plt.plot(steps1, energy22/N,'--',markersize=4,label='T=2.2');
plt.plot(steps1, energy_crit/N,'--',markersize=4,label='T= 2.269');
plt.plot(steps1, energy23/N,'--',markersize=4,label='T=2.3');
plt.plot(steps1, energy24/N,'--',markersize=4,label='T=2.4');
plt.plot(steps1, energy25/N,'--',markersize=4,label='T=2.5');
plt.plot(steps1, energy26/N,'--',markersize=4,label='T=2.6');
plt.plot(steps1, energy27/N,'--',markersize=4,label='T=2.7');
plt.plot(steps1, energy28/N,'--',markersize=4,label='T=2.8');
plt.plot(steps1, energy29/N,'--',markersize=4,label='T=2.9');
plt.plot(steps1, energy30/N,'--',markersize=4,label='T=3.0');
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
#
fig.savefig('E_vs_steps10e3.png')
plt.clf()

fig=plt.figure(12,(12,10))
#
plt.title('Magnetization time series',fontsize = 16)
plt.xlabel('MCS',fontsize=16)
plt.ylabel(r'$M/N$',fontsize=16)
plt.xlim(0,1000)


plt.plot(steps1, M20/N,'--',markersize=4,label='T=2.0');
plt.plot(steps1, M21/N,'--',markersize=4,label='T=2.1');
plt.plot(steps1, M22/N,'--',markersize=4,label='T=2.2');
plt.plot(steps1, M_crit/N,'--',markersize=4,label='T= 2.269');
#plt.plot(steps1, M23/N,'-', color='green',markersize=4,label='T=2.3');
#plt.plot(steps1, M24/N,'-',markersize=4,label='T=2.4');
#plt.plot(steps1, M25/N,'-',markersize=4,label='T=2.5');
#plt.plot(steps1, M26/N,'-',markersize=4,label='T=2.6');
#plt.plot(steps1, M27/N,'-',markersize=4,label='T=2.7');
#plt.plot(steps1, M28/N,'-',markersize=4,label='T=2.8');
#plt.plot(steps1, M29/N,'-',markersize=4,label='T=2.9');
plt.plot(steps1, M30/N,'--',markersize=4,label='T=3.0');
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
#
fig.savefig('M_vs_steps10e3.png')
plt.clf()

steps1 = steps[0::1500]
energy20 = energy1[0::1500]
energy21 = energy2[0::1500]
energy22 = energy3[0::1500]
energy23 = energy4[0::1500]
energy24 = energy5[0::1500]
energy25 = energy6[0::1500]
energy26 = energy7[0::1500]
energy27 = energy8[0::1500]
energy28 = energy9[0::1500]
energy29 = energy10[0::1500]
energy30 = energy11[0::1500]
energy_crit = energy12[0::1500]

M20 = M1[0::1500]
M21 = M2[0::1500]
M22 = M3[0::1500]
M23 = M4[0::1500]
M24 = M5[0::1500]
M25 = M6[0::1500]
M26 = M7[0::1500]
M27 = M8[0::1500]
M28 = M9[0::1500]
M29 = M10[0::1500]
M30 = M11[0::1500]
M_crit = M12[0::1500]


fig=plt.figure(12,(12,10))
#
plt.title('Energy time series',fontsize = 16)
plt.xlabel('MCS',fontsize=16)
plt.ylabel(r'$E/N$',fontsize=16)
plt.xlim(0,2000000)


plt.plot(steps1, energy20/N,'--', markersize=4,markevery=500,label='T=2.0');
plt.plot(steps1, energy21/N,'--', markersize=4,markevery=500,label='T=2.1');
plt.plot(steps1, energy22/N,'--',markersize=4,markevery=100,label='T=2.2');
plt.plot(steps1, energy_crit/N,'--',markersize=4,markevery=500,label='T= 2.269');
plt.plot(steps1, energy23/N,'--',markersize=4,label='T=2.3');
plt.plot(steps1, energy24/N,'--',markersize=4,markevery=100,label='T=2.4');
plt.plot(steps1, energy25/N,'--',markersize=4,markevery=500,label='T=2.5');
plt.plot(steps1, energy26/N,'--',markersize=4,markevery=100,label='T=2.6');
plt.plot(steps1, energy27/N,'--',markersize=4,markevery=100,label='T=2.7');
plt.plot(steps1, energy28/N,'--',markersize=4,markevery=100,label='T=2.8');
plt.plot(steps1, energy29/N,'--',markersize=4,markevery=100,label='T=2.9');
plt.plot(steps1, energy30/N,'--',markersize=4,markevery=500,label='T=3.0');
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
fig.savefig('E_vs_steps10e5.png')
plt.clf()

#
fig=plt.figure(12,(12,10))
#
plt.title('Magnetization time series',fontsize = 16)
plt.xlabel('MCS',fontsize=16)
plt.ylabel(r'$M/N$',fontsize=16)
plt.xlim(0,2000000)

plt.plot(steps1, M20/N,'--',markersize=4,label='T=2.0');
plt.plot(steps1, M21/N,'--',markersize=4,label='T=2.1');
plt.plot(steps1, M22/N,'--',markersize=4,label='T=2.2');
plt.plot(steps1, M_crit/N,'--',markersize=4,label='T= 2.269');
#plt.plot(steps1, M23/N,'-', color='green',markersize=4,label='T=2.3');
#plt.plot(steps1, M24/N,'-',markersize=4,label='T=2.4');
#plt.plot(steps1, M25/N,'-',markersize=4,label='T=2.5');
#plt.plot(steps1, M26/N,'--',markersize=4,label='T=2.6');
#plt.plot(steps1, M27/N,'-',markersize=4,label='T=2.7');
#plt.plot(steps1, M28/N,'-',markersize=4,label='T=2.8');
#plt.plot(steps1, M29/N,'-',markersize=4,label='T=2.9');
plt.plot(steps1, M30/N,'--',markersize=4,label='T=3.0');
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

fig.savefig('M_vs_steps10e5.png')
plt.clf()

#fig, ax = plt.subplots(figsize=(12,6))
#ax = plt.subplot(211)
#plt.plot(steps1, energy20/N,label='T=2.0')
#plt.legend(loc=0)
#
#ax = plt.subplot(212)
#plt.plot(steps1, energy_crit/N,label='T= 2.269',color='g')
#plt.legend(loc=0)
#plt.show()

#fig=plt.figure(10,(10,8))
#stepss = steps[8000000:]
#plt.title('Auto correlation function of E vs time')
#plt.xlabel('nMCS')
#plt.ylabel('C(E)')
#plt.plot(stepss, corr,'--', color='black',markersize=7);
#
#fig.savefig('corr.png')
#plt.clf()
