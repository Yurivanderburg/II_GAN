import numpy as np
import matplotlib.pyplot as pl
from functions_LST import create_baseline_image
import cv2
import os

# ------------------- Parameters ---------------------
observing_time = 3  # hours
light_source = [8.868, 19.846 * np.pi/180]  # [declination, hour_angle] of the source: Altair
image_size = 128  # px
PATH_out = "Data/LST-1/mask/"

# Relative positions of the submirrors: The center of the 'missing' mirror facet is the point (0,0) of the coordinate system; and we have the dimensions:
d = 1.5 #m, flat-flat minimal diameter
D = np.round((2/np.sqrt(3))*d, 1) #m, maximal diameter
t = 0.5*D #m, side length -> Equals maximal diameter/2

positions = [[0, 3*t + 3*D, 0], [3*d, 2*t + 2*D, 0], [4*d, 0, 0], [2.5*d, -2.5*D - 2.5*t, 0], [-2.5*d, -2.5*D - 2.5*t, 0], [-5*d, 0, 0], [-4*d, 2*D + 2*t, 0]]
image_name = f"{PATH_out}mask"
pl.rcParams["figure.figsize"] = (6, 6)


def main():
    """
    Generates the sparse sampling mask based on the Parameters given.
    The positions correspond to MAGIC (first two) and LST telescopes.
    Output: .npx array and .png image
    """

    # Create directory if needed
    if not os.path.exists(PATH_out):
        os.makedirs(PATH_out)
        print(f"{PATH_out} created.")

    # Create image of the baselines
    create_baseline_image(positions, observing_time, light_source, image_name)
    #pl.pause(0.5)
    mask_ = pl.imread(str(image_name + ".png"))

    # Convert to grayscale and re-arrange pixels
    mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask < 1, 1, 0).astype(np.float32)
    mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # gives the best results
    mask = np.where(mask > 0, 1, 0)

    # Save mask as numpy array
    np.save(image_name, mask)
    print("Sampling mask successfully saved.")


if __name__ == "__main__":
    main()
