import numpy as np
from aperture import aper
from obstime import observe
from obspoint import vary_base
import matplotlib.pyplot as plt
import cv2

ap = aper()
obs = observe()
vb = vary_base()

# observation time for Intensity Interferometry
start = [2460311.20167, 2460312.20167, 2460313.20167]   # (Jan 1-2, 2024, 7pm)
end = [2460311.70833, 2460312.70833, 2460313.70833]     # (Jan 2-3, 2023, 5am)
step = obs.obslen(start, end, 0.0104167)
jd = obs.julday(start, end, step)

# telescope position, name, baseline name and baseline length
# position of telescope on earth, Hanle Ladhakh
la = vb.raddeg(32, 46, 48) 
lo = vb.raddeg(78, 58, 35)

# coordinates for telescope
x = np.linspace(-60, 150, 500)
y = np.linspace(-60, 80, 500)
Tname = ['T1', 'T2', 'T3', 'T4'] 
Tpos = [135 - 15j, 40 - 50j, 30 + 60j, -40 + 10j]
fname = 'telescope.png'
ap.telescope(Tpos, Tname, fname, x, y, 6)

# baseline name and length (x, y)
tel, base = ap.baseline(Tpos, Tname) 
x = np.real(base) 
y = np.imag(base)
z = 1e-6

# position of source Polaris (assumed) in sky coordinate
r = vb.radhr(2, 31, 49.09)
de = vb.raddeg(89, 15, 50.8)

# variational baseline according to the earth rotation
vb.ps_para(lat=la, lon=lo, rac=r, dec=de)
dist = []                    # the baseline position according to earth rotation
for i in range(len(x)):
    dist.append(vb.rotbase(x[i], y[i], z, jd))

distance = np.array(dist)
xt = distance[:,0]           # baselines in east direction
yt = distance[:,1]           # baselines in north direction
zt = distance[:,2]           # baseline in up


# plot the covered (u,v) plane with baselines
plt.close()
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = [14,12]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
st = np.cumsum(step)
for k in range(0, len(xt), 1):
    plt.plot(xt[k,:], yt[k,:], '.', color="black")
    plt.axis('off')
    plt.gca().set_aspect('equal')
plt.savefig('base/baseline.png', bbox_inches='tight', pad_inches=0, dpi=500)


# Convert the observational plane to grayscale and re-arrange pixels and save the baseline in .npy format
base = plt.imread("base/baseline.png")
image_size = 128  # px
mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
mask = np.where(mask < 1, 1, 0).astype(np.float32)                                    # Return elements chosen from x or y depending on condition (condition, x, y)
mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # gives the best results
mask = np.where(mask > 0, 1, 0)
np.save("base_npy/base", mask)                                                        # Save mask as numpy array

