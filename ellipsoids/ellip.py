import numpy as np
import matplotlib.pyplot as pl

pl.style.use('dark_background')

x = np.linspace(-1,1,512)
X,Y = np.meshgrid(x,x)
T = 0*X

sq = 0.6
inc = np.pi/3
pa = np.pi/180*80
cosI, sinI = np.cos(inc), np.sin(inc)
ph = np.linspace(0,2*np.pi,101)
#for th in np.linspace(0,np.pi,11):
for c in range(-5,6):
    th = -np.pi*(c+6)/12
    print(c,th)
    cs, sn = np.cos(th), np.sin(th)
    z = sinI * cs * sq
    x = sn * np.cos(ph)
    y = sn * np.sin(ph) * cosI + z
    cs, sn = np.cos(pa), np.sin(pa)
    x,y = cs*x - sn*y, sn*x + cs*y
    if c < 0:
        col = 'blue'
    elif c == 0:
        col = 'lightblue'
    else:
        col = 'cyan'
    pl.plot(x,y,color=col)
    T[X**2 + ((Y-z)/cosI)**2 < sn**2] = th


ax = pl.gca()
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
pl.show()
