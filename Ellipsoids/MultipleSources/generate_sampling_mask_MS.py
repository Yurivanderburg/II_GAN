import numpy as np
import matplotlib.pyplot as pl
from functions import create_baseline_image
import cv2
import os

# ------------------- Parameters ---------------------
N_tele = 4  # Number of telescopes
observing_time = 3  # hours
# Source: Adhara
light_source = [-28.97, 6.977 * np.pi/180]  # [declination, hour_angle] of the source
image_size = 128  # px
PATH_out = f"Data/masks_Adhara/{image_size}px/"

# Relative positions of the telescopes:
positions_all = [[0, 0, 0], [52, 64, 0], [-92, 12, 0], [-80, 120, 0], [-196, 160, 0], [-216, 52, 0]]
positions = positions_all[:N_tele]
image_name = f"{PATH_out}MASK_{N_tele}_telescopes"
pl.rcParams["figure.figsize"] = (6, 6)


def main(source, name):
    """
    Generates the sparse sampling mask based on the Parameters given.
    The positions correspond to MAGIC (first two) and LST telescopes.
    Output: .npx array and .png image
    """
    
    PATH_out = f"Data/MultipleSources/masks/"
    image_name = f"{PATH_out}MASK_{N_tele}_telescopes_{name}"


    # Create directory if needed
    if not os.path.exists(PATH_out):
        os.makedirs(PATH_out)
        print(f"{PATH_out} created.")

    # Create image of the baselines
    create_baseline_image(positions, observing_time, source, image_name)
    pl.pause(0.5)
    mask_ = pl.imread(str(image_name + ".png"))
    """
    # Cut borders:
    if N_tele == 1:
        raise SyntaxError("There must be at least two telescopes to calculate the baseline.")
    if N_tele == 2:
        mask = mask_[950: 2070, 400: 2650]
    elif N_tele == 3:
        mask = mask_[570:2450,400:2680]
    elif N_tele == 4:
        mask = mask_[400:2600, 400:2600]
    elif N_tele == 5:
        mask = mask_[400:2650,400:2650]
    elif N_tele == 6:
        mask = mask_[390:2630, 420:2660]
    else:
        raise NotImplementedError("Please use a different number of telescopes for the calculation. Alternatively, "
                                  "you can implement it yourself by removing the border appropriately.")
   """
    # Convert to grayscale and re-arrange pixels
    mask = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
    mask = np.where(mask < 1, 1, 0).astype(np.float32)
    mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA) # gives the best results
    mask = np.where(mask > 0, 1, 0)

    # Save mask as numpy array
    np.save(image_name, mask)
    print("Sampling mask successfully saved.")


if __name__ == "__main__":
    
    sources_names = ["Altair", "Adhara", "Sirius", "Benetnasch", "Mirzam"] 
    sources = np.array([[8.868, 19.846], [28.97, 8.977], [16.716, 6.753], [49.31, 13.792], [17.956, 6.378]])
    sources[:,1] = sources[:,1] * (np.pi/180)
    
    for i in range(len(sources)):
        main(sources[i], sources_names[i])

