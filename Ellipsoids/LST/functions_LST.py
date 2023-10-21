import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as pl
import matplotlib.colors as colors
import cv2

mas = 1e-3/(180/np.pi*3600)


# -------------------------- Functions used in generate_ellipsoids.py -----------------------------
# ds is the pixel size
def grids(ds,N,lam):
    dx = lam/(N*ds)
    sx = (np.arange(N) - N//2) * ds
    sx,sy = np.meshgrid(sx,sx)
    x = (np.arange(N) - N//2) * dx
    x,y = np.meshgrid(x,x)
    return sx,sy,x,y


# Draws contour map of f (on sky or ground, directed)
# zoom factor (power of 2 preferred)
def draw(xc, yc, f, zoom, where, cmap='Greys_r', ceil=None, fceil=None, title=None):
    def cen(f):
        N = f.shape[0]
        M = N // (zoom * 2)
        return f[N // 2 - M:N // 2 + M, N // 2 - M:N // 2 + M]

    f = cen(f)
    pl.clf()
    #pl.tight_layout()
    fmax = f.max()
    fmin = f.min()
    if where == 'sky':
        sx, sy = cen(xc) / mas, cen(yc) / mas
        if ceil:
            fmin, fmax = 0, max(ceil, f.max())
        levs = np.linspace(fmin, fmax, 40)
        cs = pl.contourf(sx, sy, f, levs, cmap=cmap)
        pl.xlabel('mas')
    if where == 'ground':
        if xc[-1, -1] > 3e4:
            x, y = 1e-3 * cen(xc), 1e-3 * cen(yc)
            pl.xlabel('kilometres')
        elif xc[-1, -1] < 1:
            x, y = 1e3 * cen(xc), 1e3 * cen(yc)
            pl.xlabel('millimetres')
        else:
            x, y = cen(xc), cen(yc)
            pl.xlabel('metres')
        if ceil:
            fmin, fmax = 0, max(ceil, f.max())
            levs = np.linspace(fmin, fmax, 20)
        elif fceil:
            fmax = max(fceil, f.max())
            fmin = -fmax
            levs = np.linspace(fmin, fmax, 80)
        else:
            fmin, fmax = 0, f.max()
            levs = np.linspace(fmin, fmax, 20)
        cs = pl.contourf(x, y, f, levs, norm=colors.Normalize(vmin=fmin, vmax=fmax), cmap=cmap)
    if fmax > 10:
        fms = '%i'
    else:
        lgf = np.log10(fmax)
        ip = int(-lgf) + 2
        if lgf < -5:
            fms = '%7.1e'
        else:
            fms = '%' + '.%i' % ip + 'f'
    pl.colorbar(cs) #,format=fms)
    if title:
        pl.title(title)
    pl.gca().set_aspect('equal')


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
            draw(sx,sy,Tv,1,'sky',ceil=1,cmap='inferno')
        #pl.pause(.05)
    return Tv


# -------------------------- Functions used in generate_sampling_mask.py -----------------------------

def baseline_rotation(baseline, h, source, lamda = 1):
    """
    Calculate in the Fourier-plane due to earths rotation for a stellar light-source.
    The latitude is at La Palma, 28° 45' 25.79" N
    """

    d = source[0]
    h0 = source[1] # Hour angle -> Rad
    l = 28.757 * pi / 180  # La Palma: 28° 45' 25.79" N = +28.757

    R_d = np.array([[1, 0, 0], [0, cos(d), -sin(d)], [0, sin(d), cos(d)]])
    R_l = np.array([[1, 0, 0], [0, cos(l), sin(l)], [0, -sin(l), cos(l)]])

    # Observe for the duration h
    h_ = np.arange((h0 - h / 2), (h0 + h / 2), 0.1) * pi/12

    result = []
    for i in range(len(h_)):
        R_h = np.array([[cos(h_[i]), 0, sin(h_[i])], [0, 1, 0], [-sin(h_[i]), 0, cos(h_[i])]])

        matrix_product = (R_d @ R_h @ R_l)
        fourier_plane = np.matmul(matrix_product, baseline)
        result.append((1/lamda)*fourier_plane)
        #result.append(fourier_plane)

    return np.array(result)


def create_baseline_image(pos, time, source, image_name, lamda = 1):
    for i in range(len(pos)):
        for j in range(i):
            baseline = np.array(pos[i]) - np.array(pos[j])
            uv_plane = baseline_rotation(baseline, time, source)
            pl.plot(uv_plane[:, 0], uv_plane[:, 1], color="black")
    pl.axis('off')
    pl.gca().set_aspect('equal')
    pl.axis('square')
    pl.savefig(str(image_name+".png"), bbox_inches='tight', pad_inches=0, dpi=500)
    pl.close()
    return None


# -------------------------- Functions used in ellipsoids_calc_ff2d.py -----------------------------

# Try to concat images
def concat_images(img_a, img_b):
    """
    Combines two color image nd_arrays side-by-side.
    """
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha, :wa] = img_a
    new_img[:hb, wa:wa+wb] = img_b

    return new_img


def sap_noise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 1
    else:
        raise ValueError('This image has multiple channels, which is not supported.')

    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white

    return output


def make_grid(N, grid_size):
    xlist = []
    positions = []

    for i in range(N+1):
        if i != 0:
            xlist.append(np.round((grid_size*(i/(N+1)))))
    for i in xlist:
        for j in xlist:
            positions.append([i,j])

    return np.array(positions)


def sparse_sampling(n_ellipses, image_size, scaling=8):
    """
    Adds a mask (0 or 1s) where the image is sampled. Can be either one centered ellipse, or a grid of ellipses
    n_ellipses > 1: creates a nxn grid, so n_ellipses^2 samples
    scaling: scales the image, such that the borders are smoother (default: 64 > 512px)
    NOT USED ANYMORE!!!
    """
    img_size = image_size * scaling
    mask = np.zeros(shape=(img_size, img_size))
    angle = 135
    startAngle = 0
    endAngle = 360
    color = 1
    thickness = -1

    if n_ellipses == 1:
        center = (int(img_size / 2), int(img_size / 2))
        axesLength = (int(img_size / 8), int(img_size / 16))

        mask = cv2.ellipse(mask, center, axesLength, angle, startAngle, endAngle, color, thickness)
        mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        return mask

    else:
        mask_raw = np.zeros(shape=(img_size, img_size))
        mask_final = np.zeros(shape=(img_size, img_size))
        axesLength = (int(img_size / (3 * n_ellipses)), int(img_size / (5 * n_ellipses)))

        points = make_grid(n_ellipses, img_size)

        for i in range(len(points)):
            center = (int(points[i, 0]), int(points[i, 1]))
            # Add current ellipse
            mask_final += cv2.ellipse(mask_raw, center, axesLength, angle, startAngle, endAngle, color, thickness)
            mask_raw = np.zeros(shape=(img_size, img_size))

        mask_final = cv2.resize(mask_final, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

        return mask_final
