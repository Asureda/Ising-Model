# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Tue Jun 23 01:55:33 2020

@author: asure
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science'])
scan = np.loadtxt('ENERGY.dat')
scan1 = np.loadtxt('energy_L16.dat')
scan2 = np.loadtxt('energy_L32.dat')

Temp = scan[:,0]/2.269
Energ = scan[:,1]
Energ_L16 = scan1[:,1]
Energ_L32 = scan2[:,1]

en_error = scan[:,2]
en_error_16 = scan1[:,2]
en_error_32 = scan2[:,2]

#
fig=plt.figure(10,(10,8))



plt.title(r'$<E>/N$ vs $T/T_c$',fontsize=16)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$<E>/N$',fontsize = 16)
plt.errorbar(Temp, Energ, yerr=en_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1,label='L=100');
plt.errorbar(Temp, Energ_L32, yerr=en_error_32, fmt='-.',color='lightgreen',
             markersize=1, ecolor='green',elinewidth=1, capsize=1,label='L=32');

plt.errorbar(Temp, Energ_L16, yerr=en_error_16, fmt='-.',color='lightblue',
             markersize=1, ecolor='blue',elinewidth=1, capsize=1,label='L=16');

             
plt.legend(loc='best',fontsize=15)
plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('E_vs_T.png')
plt.show()
plt.clf()
###########################################################################
scan = np.loadtxt('MAGNETIZATION.dat')
scan1 = np.loadtxt('magnetization_L16.dat')
scan2 = np.loadtxt('magnetization_L32.dat')


Temp = scan[:,0]/2.269
Mag = scan[:,1]
Mag_error = scan[:,2]
Mag_L16 = scan1[:,1]
Mag_error_16 = scan1[:,2]
Mag_L32 = scan2[:,1]
Mag_error_32 = scan2[:,2]

#
fig=plt.figure(10,(10,8))



plt.title(r'$<|M|>/N$ vs $T/T_c$',fontsize=16)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$<|M|>/N$',fontsize = 16)
plt.errorbar(Temp, Mag, yerr=Mag_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1,label='L=100');
plt.errorbar(Temp, Mag_L32, yerr=Mag_error_32, fmt='-.',color='lightgreen',
             markersize=1, ecolor='green',elinewidth=1, capsize=1,label='L=32');

plt.errorbar(Temp, Mag_L16, yerr=Mag_error_16, fmt='-.',color='lightblue',
             markersize=1, ecolor='blue',elinewidth=1, capsize=1,label='L=16');
plt.legend(loc='best',fontsize=15)

plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('M_vs_T.png')
plt.show()
plt.clf()
############################################################################
scan = np.loadtxt('heat.dat')
scan1 = np.loadtxt('heat_L16.dat')
scan2 = np.loadtxt('heat_L32.dat')

Temp = scan[:,0]/2.269
Energ = scan[:,1]
Energ_L16 = scan1[:,1]
Energ_L32 = scan2[:,1]

en_error = scan[:,2]
en_error_16 = scan1[:,2]
en_error_32 = scan2[:,2]
#
fig=plt.figure(10,(10,8))



plt.title(r'$c$ vs $T/T_c$',fontsize=18)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$c$',fontsize = 16)
plt.text(1.1, 2, r'$c = \frac{\beta^2}{N}(<E^{2}>-<E>^{2})$', color="black", fontsize=20)
plt.errorbar(Temp, Energ, yerr=en_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1,label='L=100');
plt.errorbar(Temp, Energ_L32, yerr=en_error_32, fmt='-.',color='lightgreen',
             markersize=1, ecolor='green',elinewidth=1, capsize=1,label='L=32');

plt.errorbar(Temp, Energ_L16, yerr=en_error_16, fmt='-.',color='lightblue',
             markersize=1, ecolor='blue',elinewidth=1, capsize=1,label='L=16');
plt.legend(loc='best',fontsize=15)
        
plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('heat_vs_T.png')
plt.show()
plt.clf()
#################################################################################
scan = np.loadtxt('chi.dat')
scan1 = np.loadtxt('chi_L16.dat')
scan2 = np.loadtxt('chi_L32.dat')


Temp = scan[:,0]/2.269
Mag = scan[:,1]
Mag_error = scan[:,2]
Mag_L16 = scan1[:,1]
Mag_error_16 = scan1[:,2]
Mag_L32 = scan2[:,1]
Mag_error_32 = scan2[:,2]

#
fig=plt.figure(10,(10,8))



plt.title(r'$\chi$ vs $T/T_c$',fontsize=18)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$\chi$',fontsize = 16)
plt.text(1.1,110 , r'$\chi = \frac{\beta}{N}(<M^{2}>-<|M|>^{2})$', color="black", fontsize=20)
plt.errorbar(Temp, Mag, yerr=Mag_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1,label='L=100');
plt.errorbar(Temp, Mag_L32, yerr=Mag_error_32, fmt='-.',color='lightgreen',
             markersize=1, ecolor='green',elinewidth=1, capsize=1,label='L=32');

plt.errorbar(Temp, Mag_L16, yerr=Mag_error_16, fmt='-.',color='lightblue',
             markersize=1, ecolor='blue',elinewidth=1, capsize=1,label='L=16');
plt.legend(loc='best',fontsize=15)

plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('chi_vs_T.png')
plt.show()
plt.clf()

scan = np.loadtxt('binder_cumulant.dat')
scan1 = np.loadtxt('binder_cumulant_L16.dat')
scan2 = np.loadtxt('binder_cumulant_L32.dat')

Temp = scan[:,0]/2.269
binder = scan[:,1]
binder_error = scan[:,2]
binder_L16 = scan1[:,1]
binder_error_16 = scan1[:,2]
binder_L32 = scan2[:,1]
binder_error_32 = scan2[:,2]

fig=plt.figure(10,(10,8))

plt.title(r'$g_M$ vs $T/T_c$',fontsize=18)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$g_M$',fontsize = 16)
plt.text(1.1,0.8 , r'$g_M = \frac{1}{2}(3-\frac{<M^{4}>}{<M^{2}>^{2}})$',
         color="black", fontsize=20)
plt.errorbar(Temp, binder, yerr=binder_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1,label='100');
             
plt.errorbar(Temp, binder_L32, yerr=binder_error_32, fmt='-.',color='lightgreen',
             markersize=1, ecolor='green',elinewidth=1, capsize=1,label='32');

plt.errorbar(Temp, binder_L16, yerr=binder_error_16, fmt='-.',color='lightblue',
             markersize=1, ecolor='blue',elinewidth=1, capsize=1,label='16');
plt.legend(loc='best',fontsize=15)
             
plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('binder_vs_T.png')
plt.show()
plt.clf()

scan = np.loadtxt('binderE_cumulant.dat')

Temp = scan[:,0]/2.269
binder = scan[:,1]
binder_error = scan[:,2]

fig=plt.figure(10,(10,8))

plt.title(r'$g_E$ vs $T/T_c$',fontsize=18)
plt.xlabel(r'$T/T_c$',fontsize = 16)
plt.ylabel(r'$g_E$',fontsize = 16)
plt.text(1.1,0.9996 , r'$g_E = \frac{1}{2}(3-\frac{<E^{4}>}{<E^{2}>^{2}})$',
         color="black", fontsize=20)
plt.errorbar(Temp, binder, yerr=binder_error, fmt='-.',color='lightgrey',
             markersize=1, ecolor='black',elinewidth=1, capsize=1);
plt.xlim(2.0/2.27,3.0/2.269)

fig.savefig('binderE_vs_T.png')
plt.show()
plt.clf()


#scan = np.loadtxt('heat_binning.dat')
#
#Temp = scan[:,0]/2.269
#Energ = scan[:,1]
#en_error = scan[:,2]
##
#fig=plt.figure(10,(10,8))
#
#
#
#plt.title(r'$c$ vs $T/T_c$',fontsize=18)
#plt.xlabel(r'$T/T_c$',fontsize = 16)
#plt.ylabel(r'$c$',fontsize = 16)
#plt.text(1.1,0.8, r'$c = \frac{\beta^2}{N}(<E^{2}>-<E>^{2})$', color="black", fontsize=20)
#plt.errorbar(Temp, Energ, yerr=en_error, fmt='-.',color='lightgrey',
#             markersize=1, ecolor='black',elinewidth=1, capsize=1);
#plt.xlim(2.0/2.27,3.0/2.269)
#
#fig.savefig('heat_binning_vs_T.png')
#plt.show()
#plt.clf()
#
#scan = np.loadtxt('chi_binning.dat')
#
#Temp = scan[:,0]/2.269
#Mag = scan[:,1]
#Mag_error = scan[:,2]
#
##
#fig=plt.figure(10,(10,8))
#
#
#
#plt.title(r'$\chi$ vs $T/T_c$',fontsize=16)
#plt.xlabel(r'$T/T_c$',fontsize = 16)
#plt.ylabel(r'$\chi$',fontsize = 16)
#plt.text(1.1,1 , r'$\chi = \frac{\beta}{N}(<M^{2}>-<|M|>^{2})$', color="black", fontsize=20)
#plt.errorbar(Temp, Mag, yerr=Mag_error, fmt='-.',color='lightgrey',
#             markersize=1, ecolor='black',elinewidth=1, capsize=1);
#plt.xlim(2.0/2.27,3.0/2.269)
#
#fig.savefig('chi_binning_vs_T.png')
#plt.show()
#plt.clf()
#
