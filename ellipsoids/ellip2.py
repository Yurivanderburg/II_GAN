import numpy as np
import matplotlib.pyplot as pl
from fourier import grids
from graphics import draw

def ellip(sx,sy,rad,inc,pa,sq):
    cs,sn = np.cos(pa),np.sin(pa)
    x,y = cs*sx + sn*sy, -sn*sx + cs*sy
    cosI, sinI = np.cos(inc), np.sin(inc)
    Tv = 0*x
    for th in np.linspace(0,np.pi,101):
        cs, sn = np.cos(th), np.sin(th)
        z = sinI * cs * sq * rad
        Tv[x**2 + ((y-z)/cosI)**2 < (sn*rad)**2] = (4 + abs(cs))/5
        if np.min(Tv) < np.max(Tv):
            draw(sx,sy,Tv,2,'sky',ceil=1,cmap='inferno')
        pl.pause(.05)
    return Tv

sx,sy,x,y = grids(1e-10,512,1e-6)
ellip(sx,sy,1e-8,1.5,1,0.5)
